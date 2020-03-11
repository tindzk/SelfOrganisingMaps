#include <opencv2/opencv.hpp>
#include <morph/display.h>
#include <morph/tools.h>
#include <morph/HexGrid.h>
#include <morph/ReadCurves.h>
#include <morph/RD_Base.h>
#include <morph/RD_Plot.h>

using namespace cv;
using namespace morph;

using morph::RD_Plot;

/**
  * High-level wrapper for specifying a network so that a simulation can be built by calling the methods (e.g.,
  * step/map) in a given order.
  */
class Network {
public:
    int time;

    Network() {
        time = 0;
    };

    virtual void init(Json::Value) {

    }
};

class Square {
public:
    double x, y;
    double X;

    Square(double x, double y) {
        this->x = x;
        this->y = y;
        X = 0.;
    }
};

vector<Square> squaresFromHexGrid(const HexGrid* hexGrid) {
    vector<Square> squares;
    for (auto &v: hexGrid->vhexen) squares.push_back(Square(v->x, v->y));
    return squares;
}

/**
 * Initialise connections between `src` and `dst`
 *
 * Based on spatial distances between the source and target sheet, determine which units to connect. Then, calculate the
 * corresponding weights.
 *
 * Implements eq. 9
 */
template<class Flt>
void initialiseConnections(
        const vector<Square>& src,
        const vector<Square>& dst,
        Flt radius,
        Flt sigma,
        vector<size_t>& counts,
        vector<vector<size_t>>& srcId,
        vector<vector<Flt>>& weights
) {
    auto nSrc = src.size();
    auto nDst = dst.size();

    counts.resize(nDst);
    srcId.resize(nDst);
    weights.resize(nDst);

#pragma omp parallel for default(none) shared(nSrc) shared(nDst) shared(sigma) shared(weights) shared(src) shared(dst) shared(srcId) shared(counts) shared(radius)
    for (size_t i = 0; i < nDst; i++) {
        for (size_t j = 0; j < nSrc; j++) {
            Flt dx = src[j].x - dst[i].x;
            Flt dy = src[j].y - dst[i].y;

            Flt distSquared = dx * dx + dy * dy;

            if (distSquared < radius * radius) {
                counts[i]++;
                srcId[i].push_back(j);

                // TODO add u from eq. 9
                weights[i].push_back((sigma <= 0.) ? 1. : exp(-distSquared / (2. * sigma * sigma)));
            }
        }

        for (size_t j = 0; j < counts[i]; j++) weights[i][j] /= counts[i];
    }
}

/**
 * A projection class for connecting units on a source sheet to units on a destination sheet with topographically
 * aligned weighted connections from a radius of units on the source sheet to each destination sheet unit.
 */
template<class Flt>
class Projection {
private:
    HexGrid *hgSrc;
    HexGrid *hgDst;
public:
    Flt radius;
    Flt strength;                   // strength of projection - multiplication after dot products
    Flt alpha;                      // learning rate
    Flt sigma;
    bool normaliseAlphas;

    Flt* Xsrc;  // activations of source sheet

    vector<size_t> counts;                // number of connections in connection field for each unit
    vector<Flt> alphas;                // learning rates for each unit may depend on e.g., the number of connections
    vector<vector<size_t> > srcId;            // identity of connected units on the source sheet
    vector<vector<Flt> > weights;        // connection weights

    /**
     * Initialise the class with random weights (if sigma>0, the weights have a Gaussian pattern, else uniform random)
     *
     * @param radius radius within which connections are made
     * @param normaliseAlphas whether to normalise learning rate by individual unit connection density
     */
    Projection(Flt* Xsrc, HexGrid *hgSrc, HexGrid *hgDst, Flt radius, Flt strength, Flt alpha, Flt sigma, bool normaliseAlphas) {
        this->Xsrc = Xsrc;
        this->hgSrc = hgSrc;
        this->hgDst = hgDst;
        this->radius = radius;
        this->strength = strength;
        this->alpha = alpha;
        this->sigma = sigma;
        this->normaliseAlphas = normaliseAlphas;
    }

    void initialise() {
        initialiseConnections(squaresFromHexGrid(hgSrc), squaresFromHexGrid(hgDst), radius, sigma, counts, srcId, weights);

        auto nDst = hgDst->vhexen.size();
        alphas.resize(nDst);
#pragma omp parallel for default(none) shared(nDst)
        for (size_t i = 0; i < nDst; i++)
            alphas[i] = !normaliseAlphas ? alpha : alpha / counts[i];
    }

    vector<Flt> getWeights() {
        vector<Flt> weightStore;
        for (size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < counts[i]; j++) {
                weightStore.push_back(weights[i][j]);
            }
        }
        return weightStore;
    }

    void setWeights(vector<Flt> weightStore) {
        int k = 0;
        for (size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < counts[i]; j++) {
                weights[i][j] = weightStore[k];
                k++;
            }
        }
    }

    vector<double> getWeightPlot(int i) {
        vector<double> weightPlot;
        weightPlot.resize(hgSrc->vhexen.size());
#pragma omp parallel for
        for (size_t j = 0; j < weightPlot.size(); j++) {
            weightPlot[j] = 0.;
        }
#pragma omp parallel for
        for (size_t j = 0; j < counts[i]; j++) {
            weightPlot[srcId[i][j]] = weights[i][j];
        }
        return weightPlot;
    }
};

template<class Flt>
Flt* createVector(size_t n) {
    Flt* result = (Flt*) malloc(n * sizeof(Flt));
    memset(result, 0, sizeof(Flt) * n);
    return result;
}

template<class Flt>
class RD_Sheet : public morph::RD_Base<Flt> {
public:
    vector<Projection<Flt>> Projections;

    // thresholds
    Flt* theta = nullptr;

    // activations
    Flt* X = nullptr;

    // current activity patterns
    Flt* fields = nullptr;

    virtual void init() {
        morph::RD_Base<Flt>::allocate();

        theta = createVector<Flt>(this->nhex);
        X = createVector<Flt>(this->nhex);
    }

    ~RD_Sheet() {
        if (theta != nullptr) free(theta);
        if (X != nullptr) free(X);
        if (fields != nullptr) free(fields);
    }

    /**
     * @param sheet  source sheet
     */
    void addProjection(RD_Sheet<Flt> &sheet, float radius, float strength, float alpha, float sigma, bool normaliseAlphas) {
        Projections.push_back(Projection<Flt>(sheet.X, sheet.hg, this->hg, radius, strength, alpha, sigma, normaliseAlphas));
    }

    virtual void allocate() {
        fields = createVector<Flt>(Projections.size() * this->nhex);
        for (auto &p: Projections) p.initialise();
    }

    virtual void step() {}
};

template<class Flt>
void copyActivations(const RD_Sheet<Flt>& src, const RD_Sheet<Flt>& dst) {
    assert(src.nhex == dst.nhex);
    for (size_t i = 0; i < src.nhex; i++) dst.X[i] = src.X[i];
}

template<class Flt>
inline void zero_X(RD_Sheet<Flt>& sheet) {
#pragma omp parallel for default(none) shared(sheet)
    for (size_t hi = 0; hi < sheet.nhex; ++hi) sheet.X[hi] = 0.;
}

template<class Flt>
inline vector<Flt> activations(const RD_Sheet<Flt>& sheet) {
    vector<Flt> result;
    for (size_t i = 0; i < sheet.nhex; i++) result.push_back(sheet.X[i]);
    return result;
}

template<class Flt>
void renormalise(RD_Sheet<Flt>& sheet, const vector<size_t>& projections) {
#pragma omp parallel for default(none) shared(projections) shared(sheet)
    for (size_t i = 0; i < sheet.nhex; i++) {
        Flt sumWeights = 0.0;

        for (auto projectionId: projections) {
            auto &p = sheet.Projections[projectionId];
            for (size_t j = 0; j < p.counts[i]; j++)
                sumWeights += p.weights[i][j];
        }

        for (auto projectionId: projections) {
            auto &p = sheet.Projections[projectionId];
            for (size_t j = 0; j < p.counts[i]; j++)
                p.weights[i][j] /= sumWeights;
        }
    }
}

template<class Flt>
void sheetStep(RD_Sheet<Flt>& sheet, const vector<Projection<Flt>>& projections) {
    // Calculate activity patterns
    for (size_t pi = 0; pi < projections.size(); pi++) {
        auto& p = projections[pi];

        /* Dot product of each weight vector with the corresponding source sheet field values, multiplied by the
         * strength of the projection
         */
#pragma omp parallel for default(none) shared(sheet) shared(p) shared(pi)
        for (size_t hi = 0; hi < sheet.nhex; hi++) {
            auto field = 0.;
            for (size_t hj = 0; hj < p.counts[hi]; hj++)
                field += p.Xsrc[p.srcId[hi][hj]] * p.weights[hi][hj];
            field *= p.strength;

            sheet.fields[pi * sheet.nhex + hi] = field;
        }
    }

    // This must not be performed before calculating the activity patterns because `Xsrc` could contain self connections
    // as in the case of the cortical sheet.
#pragma omp parallel for default(none) shared(sheet)
    for (size_t hi = 0; hi < sheet.nhex; ++hi) sheet.X[hi] = 0.;

    for (size_t pi = 0; pi < projections.size(); pi++) {
#pragma omp parallel for default(none) shared(sheet) shared(pi)
        for (size_t hi = 0; hi < sheet.nhex; ++hi)
            sheet.X[hi] += sheet.fields[pi * sheet.nhex + hi];
    }

#pragma omp parallel for default(none) shared(sheet)
    for (size_t hi = 0; hi < sheet.nhex; ++hi)
        sheet.X[hi] = fmax(sheet.X[hi] - sheet.theta[hi], 0.);
}

template<class Flt>
inline void sheetStep(RD_Sheet<Flt>& sheet) {
    sheetStep(sheet, sheet.Projections);
}

template<class Flt>
inline double hebbian(
        const Projection<Flt> &p,
        const Flt* Xsrc,
        const Flt* Xdst,
        const size_t i,
        const size_t j
) {
    auto omega_ij_p = p.weights[i][j];
    auto alpha_p = p.alphas[i];
    auto eta_i = Xsrc[p.srcId[i][j]];
    auto eta_j = Xdst[i];

    return omega_ij_p + alpha_p * eta_j * eta_i;
}

/**
 * Hebbian weight adaptation
 *
 * Updates weights of supplied projections, constraining them using divisive postsynaptic weight normalisation.
 *
 * From paper: "This rule results in connections that reflect correlations between the presynaptic ON/OFF unit
 * activities and the postsynaptic V1 response."
 *
 * Implements eq. 10
 *
 * @param projections Subset of projections. In the case of V1, this allows excluding self connections.
 */
void learn(RD_Sheet<double>& sheet, const vector<size_t> projections) {
    for (auto projectionId: projections) {
        auto &p = sheet.Projections[projectionId];

#pragma omp parallel for default(none) shared(p) shared(sheet)
        for (size_t i = 0; i < sheet.nhex; i++)
            for (size_t j = 0; j < p.counts[i]; j++)
                if (p.alphas[i] > 0.0) p.weights[i][j] = hebbian(p, p.Xsrc, sheet.X, i, j);
    }

    // From paper: "All afferent connection weights from RGC/LGN sheets are normalized together in the model, which
    // allows V1 neurons to become selective for any subset of the RGC/LGN inputs."
    renormalise(sheet, projections);
}

template<class Flt>
struct HomeostasisParameters {
    // Degree of smoothing in average calculation
    Flt beta;

    // Target V1 unit activity
    Flt mu;

    // Homeostatic learning rate
    Flt lambda;

    // Initial threshold
    Flt thetaInit;
};

template<class Flt>
class CortexSOM : public RD_Sheet<Flt> {
private:
    HomeostasisParameters<Flt> params;

    // Smoothed average activities
    Flt* Xavg = nullptr;
public:
    virtual void init(const HomeostasisParameters<Flt> params) {
        RD_Sheet<Flt>::init();
        this->params = params;
    }

    virtual void allocate() {
        RD_Sheet<Flt>::allocate();
        this->Xavg = createVector<Flt>(this->nhex);

        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->Xavg[hi] = params.mu;
            this->theta[hi] = params.thetaInit;
        }
    }

    ~CortexSOM() {
        if (Xavg != nullptr) free(Xavg);
    }

    /**
     * Perform homeostasis, i.e. calculate average activations and update thresholds
     *
     * From paper: "The effect of this scaling mechanism is to bring the average activity of each V1 unit closer to the
     * specified target."
     */
    void homeostasis() {
#pragma omp parallel for default(none)
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            // Calculate average given degree of smoothing (beta) as per eq. 7
            Xavg[hi] = (1. - params.beta) * this->X[hi] + params.beta * Xavg[hi];

            // Update thresholds as per eq. 8
            this->theta[hi] += params.lambda * (Xavg[hi] - params.mu);
        }
    }
};

/** Input sheet */
template<class Flt>
class PatternGenerator_Sheet : public RD_Sheet<Flt> {
public:
    void Gaussian(double x_center, double y_center, double theta, double sigmaA, double sigmaB) {
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);
        double overSigmaA = 1. / sigmaA;
        double overSigmaB = 1. / sigmaB;
#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            Flt dx = this->hg->vhexen[hi]->x - x_center;
            Flt dy = this->hg->vhexen[hi]->y - y_center;
            this->X[hi] = exp(-((dx * cosTheta - dy * sinTheta) * (dx * cosTheta - dy * sinTheta)) * overSigmaA
                              - ((dx * sinTheta + dy * cosTheta) * (dx * sinTheta + dy * cosTheta)) * overSigmaB);
        }
    }

    void Grating(double theta, double phase, double width, double amplitude) {
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);

#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->X[hi] = sin(
                    width * (this->hg->vhexen[hi]->x * sinTheta + this->hg->vhexen[hi]->y * cosTheta + phase));
        }
    }
};


// This helper function is general-purpose and should really be moved into morphologica
vector<double> getPolyPixelVals(Mat frame, vector<Point> pp) {
    Point pts[4] = {pp[0], pp[1], pp[2], pp[3]};
    Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
    fillConvexPoly(mask, pts, 4, cv::Scalar(255, 255, 255));
    Mat result, resultGray;
    frame.copyTo(result, mask);
    cvtColor(result, resultGray, cv::COLOR_BGR2GRAY);
    vector<Point2i> positives;
    findNonZero(resultGray, positives);
    vector<double> polyPixelVals(positives.size());
    for (size_t j = 0; j < positives.size(); j++) {
        Scalar pixel = resultGray.at<uchar>(positives[j]);
        polyPixelVals[j] = (double) pixel.val[0] / 255.;
    }
    return polyPixelVals;
}

class CartGrid {
public:
    int n, nx, ny;
    vector<Square> vsquare;

    void init(int nx, int ny) {
        this->nx = nx;
        this->ny = ny;
        n = nx * ny;

        auto maxDim = (double) max(nx, ny);

        int k = 0;
        for (int i = 0; i < nx; i++) {
            double xpos = ((double) i / maxDim) - 0.5;
            for (int j = 0; j < ny; j++) {
                double ypos = ((double) j / maxDim) - 0.5;
                vsquare.push_back(Square(xpos, ypos));
                k++;
            }
        }
    }
};


template<class Flt>
class HexCartSampler : public RD_Sheet<Flt> {
public:
    CartGrid C;
    VideoCapture cap;
    vector<vector<size_t>> srcId;
    vector<vector<Flt>> weights;
    vector<size_t> counts;
    Flt strength;
    vector<Point> mask;
    size_t stepsize;

    vector<vector<double>> PreLoadedPatterns;

    void initProjection(int nx, int ny, Flt radius, Flt sigma) {
        this->strength = 1.;
        C.init(nx, ny);
        initialiseConnections(C.vsquare, squaresFromHexGrid(this->hg), radius, sigma, counts, srcId, weights);
    }

    int initCamera(int xoff, int yoff, int stepsize) {
        this->stepsize = stepsize;
        mask.resize(4);
        mask[0] = Point(xoff, yoff);
        mask[1] = Point(xoff + stepsize * C.nx - 1, yoff);
        mask[2] = Point(xoff + stepsize * C.nx - 1, yoff + stepsize * C.ny - 1);
        mask[3] = Point(xoff, yoff + stepsize * C.ny - 1);
        return cap.open(0);
    }

    virtual void step() {
        zero_X(*this);
#pragma omp parallel for
        for (size_t i = 0; i < this->nhex; i++) {
            for (size_t j = 0; j < counts[i]; j++) {
                this->X[i] += C.vsquare[srcId[i][j]].X * weights[i][j];
            }
            this->X[i] *= strength;
        }
    }

    void stepCamera() {
        Mat frame;
        cap >> frame;
        vector<double> img = getPolyPixelVals(frame, mask);
        vector<double> pat((img.size() / stepsize), 0.);
        int iter = stepsize * C.ny * stepsize;
        int k = 0;
        for (int i = 0; i < C.nx; i++) {
            int I = (C.nx - i - 1) * stepsize;
            for (int j = 0; j < C.ny; j++) {
                C.vsquare[k].X = img[(C.ny - j - 1) * iter + I];
                k++;
            }
        }
        step();
    }

    void preloadPatterns(const string filename) {
        HdfData data(filename, 1);
        vector<double> tmp;
        data.read_contained_vals("P", tmp);
        int nPat = tmp.size() / C.n;
        PreLoadedPatterns.resize(nPat, vector<double>(C.n, 0.));
        int k = 0;
        for (int i = 0; i < nPat; i++) {
            for (int j = 0; j < C.n; j++) {
                PreLoadedPatterns[i][j] = tmp[k];
                k++;
            }
        }
    }

    void stepPreloaded(int p) {
        for (int i = 0; i < C.n; i++) {
            C.vsquare[i].X = PreLoadedPatterns[p][i];
        }
        step();
    }

    void stepPreloaded() {
        int p = floor(morph::Tools::randDouble() * PreLoadedPatterns.size());
        stepPreloaded(p);
    }
};