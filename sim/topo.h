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

template<class Flt>
Flt* createVector(size_t n) {
    Flt* result = (Flt*) malloc(n * sizeof(Flt));
    memset(result, 0, n * sizeof(Flt));
    return result;
}

vector<Square> squaresFromHexGrid(const HexGrid* hexGrid) {
    vector<Square> squares;
    for (auto &v: hexGrid->vhexen) squares.push_back(Square(v->x, v->y));
    return squares;
}

template<class Flt>
struct Connections {
    // identity of connected units on the source sheet
    size_t* srcId;

    // connection weights [nDst, nSrc]
    // all unused weights are 0
    Flt* weights;

    // number of connections in connection field for each unit
    size_t* counts;

    size_t nSrc;
    size_t nDst;
};

/**
 * Initialise connections between `src` and `dst`
 *
 * Based on spatial distances between the source and target sheet, determine which units to connect. Then, calculate the
 * corresponding weights.
 *
 * Implements eq. 3, but only the first term since both terms are equivalent except for their signs.
 *
 * @param radius radius within which connections are made
 */
template<class Flt>
Connections<Flt> createConnections(
        const vector<Square>& src,
        const vector<Square>& dst,
        Flt radius,
        Flt sigma
) {
    Connections<Flt> result = Connections<Flt>();

    result.nSrc = src.size();
    result.nDst = dst.size();
    result.counts = createVector<size_t>(result.nDst);
    result.srcId = createVector<size_t>(result.nDst * result.nSrc);
    result.weights = createVector<Flt>(result.nDst * result.nSrc);

#pragma omp parallel for default(none) shared(sigma) shared(result) shared(src) shared(dst) shared(radius)
    for (size_t i = 0; i < result.nDst; i++) {
        for (size_t j = 0; j < result.nSrc; j++) {
            Flt dx = src[j].x - dst[i].x;
            Flt dy = src[j].y - dst[i].y;

            Flt distSquared = dx * dx + dy * dy;

            if (distSquared < radius * radius) {
                result.weights[i * result.nSrc + result.counts[i]] =
                        // TODO should be +distSquared?
                        (sigma <= 0.) ? 1. : exp(-distSquared / (2. * sigma * sigma));
                result.srcId[i * result.nSrc + result.counts[i]] = j;
                result.counts[i]++;
            }
        }

        for (size_t j = 0; j < result.counts[i]; j++) result.weights[i * result.nSrc + j] /= result.counts[i];
    }

    return result;
}

/** Eqn. 4 */
template<class Flt>
Flt* createWeightsGainControl(
        const vector<Square>& src,
        const vector<Square>& dst,
        Flt radius,
        Flt sigma
) {
    auto weights = createVector<Flt>(dst.size() * src.size());

#pragma omp parallel for default(none) shared(sigma) shared(weights) shared(src) shared(dst) shared(radius)
    for (size_t i = 0; i < dst.size(); i++) {
        size_t count = 0;

        for (size_t j = 0; j < src.size(); j++) {
            Flt dx = src[j].x - dst[i].x;
            Flt dy = src[j].y - dst[i].y;

            Flt distSquared = dx * dx + dy * dy;
            if (distSquared < radius * radius) {
                weights[i * src.size() + count] =
                    // Note the minus sign before `distSquared`
                    (sigma <= 0.) ? 1. : exp(-distSquared / (2. * sigma * sigma));
                count++;
            }
        }

        for (size_t j = 0; j < count; j++) weights[i * src.size() + j] /= count;
    }

    return weights;
}

/**
 * A projection class for connecting units on a source sheet to units on a destination sheet with topographically
 * aligned weighted connections from a radius of units on the source sheet to each destination sheet unit.
 */
template<class Flt>
class Projection {
public:
    Flt strength;                  // strength of projection - multiplication after dot products
    Flt k;
    Flt gamma_S;
    Flt* Xsrc;                     // activations of source sheet
    vector<Flt> alphas;            // learning rates for each unit may depend on e.g., the number of connections
    Connections<Flt> connections;

    /**
     * Initialise the class with random weights (if sigma>0, the weights have a Gaussian pattern, else uniform random)
     *
     * @param alpha           learning rate
     * @param strength        multiplier for overall strength of connections
     * @param gamma_S         strength of feedforward contrast-gain control
     * @param normaliseAlphas normalise learning rate by individual unit connection density
     */
    Projection(Flt* Xsrc,
               Connections<Flt> connections,
               Flt strength,
               Flt k,
               Flt gamma_S,
               Flt alpha,
               bool normaliseAlphas
              ) {
        this->Xsrc = Xsrc;
        this->connections = connections;
        this->strength = strength;
        this->k = k;
        this->gamma_S = gamma_S;

        alphas.resize(connections.nDst);
#pragma omp parallel for default(none) shared(connections) shared(normaliseAlphas) shared(alpha)
        for (size_t i = 0; i < connections.nDst; i++)
            alphas[i] = !normaliseAlphas ? alpha : alpha / connections.counts[i];
    }

    vector<Flt> getWeights() {
        vector<Flt> weightStore;
        for (size_t i = 0; i < connections.nDst; i++) {
            for (size_t j = 0; j < connections.counts[i]; j++) {
                weightStore.push_back(connections.weights[i * connections.nSrc + j]);
            }
        }
        return weightStore;
    }

    void setWeights(vector<Flt> weightStore) {
        int k = 0;
        for (size_t i = 0; i < connections.nDst; i++) {
            for (size_t j = 0; j < connections.counts[i]; j++) {
                connections.weights[i * connections.nSrc + j] = weightStore[k];
                k++;
            }
        }
    }

    vector<double> getWeightPlot(int i) {
        vector<double> weightPlot;
        weightPlot.resize(connections.nSrc);
#pragma omp parallel for
        for (size_t j = 0; j < weightPlot.size(); j++) {
            weightPlot[j] = 0.;
        }
#pragma omp parallel for
        for (size_t j = 0; j < connections.counts[i]; j++) {
            weightPlot[connections.srcId[i * connections.nSrc + j]] =
                    connections.weights[i * connections.nSrc + j];
        }
        return weightPlot;
    }
};

HexGrid* createHexGrid(string svgPath) {
    HexGrid* hg = new HexGrid(0.01, 4, 0, morph::HexDomainShape::Boundary);

    ReadCurves r;
    r.init(svgPath);
    hg->setBoundary(r.getCorticalPath());

    hg->computeDistanceToBoundary();

    return hg;
}

template<class Flt>
class RD_Sheet {
public:
    vector<Projection<Flt>> Projections;

    // thresholds
    Flt* theta = nullptr;

    // activations
    Flt* X = nullptr;

    // current activity patterns
    Flt* fields = nullptr;

    // number of cells
    size_t nhex;

    /**
     * @param num Number of units
     */
    virtual void init(size_t num) {
        nhex = num;
        theta = createVector<Flt>(num);
        X = createVector<Flt>(num);
    }

    ~RD_Sheet() {
        if (theta != nullptr) free(theta);
        if (X != nullptr) free(X);
        if (fields != nullptr) free(fields);
    }

    void connect(const vector<Projection<Flt>>& projections) {
        Projections = projections;
        fields = createVector<Flt>(this->nhex * Projections.size());
    }
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
            for (size_t j = 0; j < p.connections.counts[i]; j++)
                sumWeights += p.connections.weights[i * p.connections.nSrc + j];
        }

        for (auto projectionId: projections) {
            auto &p = sheet.Projections[projectionId];
            for (size_t j = 0; j < p.connections.counts[i]; j++)
                p.connections.weights[i * p.connections.nSrc + j] /= sumWeights;
        }
    }
}

/**
 * Eqn. 2
 *
 * @param gainControlWeights For LGN only; obtained using createWeightsGainControl()
 */
template<class Flt>
void sheetStep(RD_Sheet<Flt>& sheet, const vector<Projection<Flt>>& projections, Flt* gainControlWeights) {
    // Calculate activity patterns
    for (size_t pi = 0; pi < projections.size(); pi++) {
        auto& p = projections[pi];

#pragma omp parallel for default(none) shared(sheet) shared(p) shared(pi) shared(gainControlWeights)
        for (size_t hi = 0; hi < sheet.nhex; hi++) {
            /* Dot product of each weight vector with the corresponding source sheet field values, multiplied by the
             * strength of the projection
             */
            auto activation = 0.;
            for (size_t hj = 0; hj < p.connections.counts[hi]; hj++) {
                auto afferentConnection = p.Xsrc[p.connections.srcId[hi * p.connections.nSrc + hj]];
                activation += afferentConnection * p.connections.weights[hi * p.connections.nSrc + hj];
            }
            activation *= p.strength;

            // normalisation term for contrast-gain control
            auto normalisation = 1.;
            if (gainControlWeights != NULL)  {
                normalisation = 0.;
                for (size_t hj = 0; hj < p.connections.counts[hi]; hj++)
                    normalisation += sheet.X[hi] * gainControlWeights[hi * p.connections.nSrc + hj];
                normalisation *= p.gamma_S;
                normalisation += p.k;
            }

            sheet.fields[pi * sheet.nhex + hi] = activation / normalisation;
        }
    }

    //
    // Determine activation by summing up all projection fields
    //

    // This must not be performed before calculating the activity patterns because `Xsrc` could contain self connections
    // as in the case of the cortical sheet.
    // Also, constrast-gain control needs to access previous activity.
#pragma omp parallel for default(none) shared(sheet)
    for (size_t hi = 0; hi < sheet.nhex; ++hi) sheet.X[hi] = 0.;

    for (size_t pi = 0; pi < projections.size(); pi++) {
#pragma omp parallel for default(none) shared(sheet) shared(pi)
        for (size_t hi = 0; hi < sheet.nhex; ++hi)
            sheet.X[hi] += sheet.fields[pi * sheet.nhex + hi];
    }

    // Corresponds to `f` in eqn. 2: half-wave rectifying activation function such that sheet.X[hi] >= 0
#pragma omp parallel for default(none) shared(sheet)
    for (size_t hi = 0; hi < sheet.nhex; ++hi)
        sheet.X[hi] = fmax(sheet.X[hi] - sheet.theta[hi], 0.);
}

template<class Flt>
inline void sheetStep(RD_Sheet<Flt>& sheet, Flt* gainControlWeights) {
    sheetStep(sheet, sheet.Projections, gainControlWeights);
}

template<class Flt>
inline double hebbian(
        const Projection<Flt> &p,
        const Flt* Xsrc,
        const Flt* Xdst,
        size_t i,
        size_t j
) {
    auto omega_ij_p = p.connections.weights[i * p.connections.nSrc + j];
    auto alpha_p = p.alphas[i];
    auto eta_i = Xsrc[p.connections.srcId[i * p.connections.nSrc + j]];
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
            for (size_t j = 0; j < p.connections.counts[i]; j++)
                if (p.alphas[i] > 0.0) p.connections.weights[i * p.connections.nSrc + j] =
                        hebbian(p, p.Xsrc, sheet.X, i, j);
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
    virtual void init(size_t num, const HomeostasisParameters<Flt> params) {
        RD_Sheet<Flt>::init(num);
        this->params = params;
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
    void Gaussian(HexGrid *hg, double x_center, double y_center, double theta, double sigmaA, double sigmaB) {
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);
        double overSigmaA = 1. / sigmaA;
        double overSigmaB = 1. / sigmaB;
#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            Flt dx = hg->vhexen[hi]->x - x_center;
            Flt dy = hg->vhexen[hi]->y - y_center;
            this->X[hi] = exp(-((dx * cosTheta - dy * sinTheta) * (dx * cosTheta - dy * sinTheta)) * overSigmaA
                              - ((dx * sinTheta + dy * cosTheta) * (dx * sinTheta + dy * cosTheta)) * overSigmaB);
        }
    }

    void Grating(HexGrid *hg, double theta, double phase, double width, double amplitude) {
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);

#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->X[hi] = sin(
                    width * (hg->vhexen[hi]->x * sinTheta + hg->vhexen[hi]->y * cosTheta + phase));
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
    Flt strength;
    vector<Point> mask;
    size_t stepsize;
    vector<vector<double>> PreLoadedPatterns;
    Connections<Flt> connections;

    void initProjection(HexGrid *hg, int nx, int ny, Flt radius, Flt sigma) {
        this->strength = 1.;
        C.init(nx, ny);
        this->connections = createConnections(C.vsquare, squaresFromHexGrid(hg), radius, sigma);
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
        for (size_t i = 0; i < connections.nDst; i++) {
            for (size_t j = 0; j < connections.counts[i]; j++) {
                this->X[i] += C.vsquare[connections.srcId[i * connections.nSrc + j]].X *
                        connections.weights[i * connections.nSrc + j];
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