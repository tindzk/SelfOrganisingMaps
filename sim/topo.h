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

/**
 * A projection class for connecting units on a source sheet to units on a destination sheet with topographically
 * aligned weighted connections from a radius of units on the source sheet to each destination sheet unit.
 */
template<class Flt>
class Projection {
public:
    Flt strength;                   // strength of projection - multiplication after dot products
    Flt alpha;                      // learning rate

    vector<Flt> *Xsrc;  // activations of source sheet

    vector<size_t> counts;                // number of connections in connection field for each unit
    vector<Flt> norms;                // 1./counts
    vector<Flt> alphas;                // learning rates for each unit may depend on e.g., the number of connections
    vector<vector<size_t> > srcId;            // identity of connected units on the source sheet
    vector<vector<Flt> > weights;        // connection weights
    vector<vector<Flt> > distances;        // pre-compute distances between units in source and destination sheets
    vector<double> weightPlot;            // for constructing activity plots

    /**
     * Initialise the class with random weights (if sigma>0, the weights have a Gaussian pattern, else uniform random)
     *
     * @param radius radius within which connections are made
     * @param normalizeAlphas whether to normalize learning rate by individual unit connection density
     */
    Projection(vector<Flt>* Xsrc, HexGrid *hgSrc, HexGrid *hgDst, Flt radius, Flt strength, Flt alpha, Flt sigma, bool normalizeAlphas) {
        this->Xsrc = Xsrc;
        this->strength = strength;
        this->alpha = alpha;

        auto nDst = hgDst->vhexen.size();
        auto nSrc = hgSrc->vhexen.size();

        counts.resize(nDst);
        norms.resize(nDst);
        srcId.resize(nDst);
        weights.resize(nDst);
        alphas.resize(nDst);
        distances.resize(nDst);
        weightPlot.resize(nSrc);

        Flt radiusSquared = radius * radius;    // precompute for speed

        double OverTwoSigmaSquared = 1. / (sigma * sigma * 2.0);    // precompute normalisation constant

        // initialize connections for each destination sheet unit
#pragma omp parallel for
        for (size_t i = 0; i < nDst; i++) {
            for (size_t j = 0; j < nSrc; j++) {
                Flt dx = (hgSrc->vhexen[j]->x - hgDst->vhexen[i]->x);
                Flt dy = (hgSrc->vhexen[j]->y - hgDst->vhexen[i]->y);
                Flt distSquared = dx * dx + dy * dy;
                if (distSquared < radiusSquared) {
                    counts[i]++;
                    srcId[i].push_back(j);
                    Flt w = 1.0;
                    if (sigma > 0.) {
                        w = exp(-distSquared * OverTwoSigmaSquared);
                    }
                    weights[i].push_back(w);
                    distances[i].push_back(sqrt(distSquared));
                }
            }
            norms[i] = 1.0 / (Flt) counts[i];
            alphas[i] = alpha;
            if (normalizeAlphas) {
                alphas[i] *= norms[i];
            }
        }
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
class RD_Sheet : public morph::RD_Base<Flt> {
protected:
    vector<Flt> fields;                // current activity patterns
public:
    vector<Projection<Flt>> Projections;
    alignas(alignof(vector<Flt>)) vector<Flt> X;

    virtual void init() {
        this->stepCount = 0;
        this->zero_vector_variable(this->X);
    }

    virtual void allocate() {
        morph::RD_Base<Flt>::allocate();
        this->resize_vector_variable(this->X);
    }

    /**
     * @param sheet  source sheet
     */
    void addProjection(RD_Sheet<Flt> &sheet, float radius, float strength, float alpha, float sigma, bool normalizeAlphas) {
        Projections.push_back(Projection<Flt>(&sheet.X, sheet.hg, this->hg, radius, strength, alpha, sigma, normalizeAlphas));
        this->fields.resize(Projections.size() * this->nhex, 0.0);
    }

    void zero_X() {
#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->X[hi] = 0.;
        }
    }
};

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

/**
 * Assumes that assert(X.size() == nhex)
 */
template<class Flt>
void sheetStep(
        size_t nhex,
        const vector<Projection<Flt>>& projections,
        const vector<Flt>& xOffsets,
        vector<Flt>& fields,
        vector<Flt>& X
) {
    // Calculate activity patterns
    for (size_t pi = 0; pi < projections.size(); pi++) {
        auto& p = projections[pi];

        /* Dot product of each weight vector with the corresponding source sheet field values, multiplied by the
         * strength of the projection
         */
#pragma omp parallel for default(none) shared(nhex) shared(fields) shared(p) shared(pi)
        for (size_t hi = 0; hi < nhex; hi++) {
            auto field = 0.;
            for (size_t hj = 0; hj < p.counts[hi]; hj++)
                field += (*p.Xsrc)[p.srcId[hi][hj]] * p.weights[hi][hj];
            field *= p.strength;

            fields[pi * nhex + hi] = field;
        }
    }

    // This must not be performed before calculating the activity patterns because `fSrc` could contain self connections
    // as in the case of the cortical sheet.
#pragma omp parallel for default(none) shared(nhex) shared(X)
    for (size_t hi = 0; hi < nhex; ++hi) X[hi] = 0.;

    for (size_t pi = 0; pi < projections.size(); pi++) {
#pragma omp parallel for default(none) shared(nhex) shared(X) shared(fields) shared(pi)
        for (size_t hi = 0; hi < nhex; ++hi) {
            X[hi] += fields[pi * nhex + hi];
        }
    }

#pragma omp parallel for default(none) shared(nhex) shared(X) shared(xOffsets)
    for (size_t hi = 0; hi < nhex; ++hi)
        X[hi] = fmax(X[hi] - xOffsets[hi], 0.);
}

inline double hebbian(
        const Projection<double> &p,
        const vector<double>& Xsrc,
        const vector<double>& Xdst,
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
                if (p.alphas[i] > 0.0) p.weights[i][j] = hebbian(p, *p.Xsrc, sheet.X, i, j);
    }

    // From paper: "All afferent connection weights from RGC/LGN sheets are normalized together in the model, which
    // allows V1 neurons to become selective for any subset of the RGC/LGN inputs."
    renormalise(sheet, projections);
}

template<class Flt>
class LGN : public RD_Sheet<Flt> {
private:
    alignas(alignof(vector<Flt>)) vector<Flt> zeros;
public:
    virtual void allocate() {
        RD_Sheet<Flt>::allocate();
        this->resize_vector_variable(this->zeros);
        this->zero_vector_variable(this->zeros);
    }
    virtual void step() {
        this->stepCount++;
        sheetStep(this->nhex, this->Projections, this->zeros, this->fields, this->X);
    }
};

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
    alignas(alignof(vector<Flt>)) vector<Flt> Xavg;

    // Thresholds
    alignas(alignof(vector<Flt>)) vector<Flt> Theta;
public:
    virtual void init(const HomeostasisParameters<Flt> params) {
        RD_Sheet<Flt>::init();
        this->params = params;
        this->zero_vector_variable(this->Xavg);
        this->zero_vector_variable(this->Theta);
    }

    virtual void allocate() {
        RD_Sheet<Flt>::allocate();
        this->resize_vector_variable(this->Xavg);
        this->resize_vector_variable(this->Theta);
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->Xavg[hi] = params.mu;
            this->Theta[hi] = params.thetaInit;
        }
    }

    virtual void step() {
        this->stepCount++;
        sheetStep(this->nhex, this->Projections, this->Theta, this->fields, this->X);
    }

    virtual void step(const vector<Projection<Flt>>& projections) {
        this->stepCount++;
        sheetStep(this->nhex, projections, this->Theta, this->fields, this->X);
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
            Theta[hi] = Theta[hi] + params.lambda * (Xavg[hi] - params.mu);
        }
    }
};

template<class Flt>
class PatternGenerator_Sheet : public RD_Sheet<Flt> {
public:
    virtual void step() {
        this->stepCount++;
    }

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
    vector<vector<size_t> > srcId;
    vector<vector<Flt> > weights;
    vector<vector<Flt> > distances;
    vector<size_t> counts;
    vector<Flt> norms;
    Flt strength;
    vector<Point> mask;
    size_t stepsize;

    vector<vector<double> > PreLoadedPatterns;

    void initProjection(int nx, int ny, Flt radius, Flt sigma) {
        C.init(nx, ny);

        this->strength = 1.0;
        srcId.resize(this->nhex);
        counts.resize(this->nhex);
        weights.resize(this->nhex);
        distances.resize(this->nhex);
        norms.resize(this->nhex);

        Flt radiusSquared = radius * radius;    // precompute for speed
        Flt OverTwoSigmaSquared = 1. / (2.0 * sigma * sigma);
#pragma omp parallel for
        for (size_t i = 0; i < this->nhex; i++) {
            for (int j = 0; j < C.n; j++) {
                Flt dx = (this->hg->vhexen[i]->x - C.vsquare[j].x);
                Flt dy = (this->hg->vhexen[i]->y - C.vsquare[j].y);
                Flt distSquared = dx * dx + dy * dy;
                if (distSquared < radiusSquared) {
                    counts[i]++;
                    srcId[i].push_back(j);
                    Flt w = 1.0;
                    if (sigma > 0.) {
                        w = exp(-distSquared * OverTwoSigmaSquared);
                    }
                    weights[i].push_back(w);
                    distances[i].push_back(sqrt(distSquared));
                }
            }
            norms[i] = 1.0 / (Flt) counts[i];
            for (size_t j = 0; j < counts[i]; j++) {
                weights[i][j] *= norms[i];
            }
        }

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
        this->zero_X();
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