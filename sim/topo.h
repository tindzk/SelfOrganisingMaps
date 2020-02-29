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

    Network(void) {
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
    size_t nSrc;              // number of units on source sheet
    size_t nDst;              // number of units on destination sheet

    vector<size_t> counts;                // number of connections in connection field for each unit
    vector<Flt> norms;                // 1./counts
    vector<Flt> alphas;                // learning rates for each unit may depend on e.g., the number of connections
    vector<vector<size_t> > srcId;            // identity of connected units on the source sheet
    vector<vector<Flt> > weights;        // connection weights
    vector<vector<Flt> > distances;        // pre-compute distances between units in source and destination sheets
    vector<Flt> field;                // current activity patterns
    vector<Flt *> fSrc;                // pointers to the field elements on the source sheet
    vector<Flt *> fDst;                // pointers to the field elements on the destination sheet
    vector<double> weightPlot;            // for constructing activity plots

    /**
     * Initialise the class with random weights (if sigma>0, the weights have a Gaussian pattern, else uniform random)
     *
     * @param radius radius within which connections are made
     * @param normalizeAlphas whether to normalize learning rate by individual unit connection density
     */
    Projection(vector<Flt *> fSrc, vector<Flt *> fDst, HexGrid *hgSrc, HexGrid *hgDst, Flt radius, Flt strength,
               Flt alpha, Flt sigma, bool normalizeAlphas) {
        this->fSrc = fSrc;
        this->fDst = fDst;
        this->strength = strength;
        this->alpha = alpha;

        nDst = hgDst->vhexen.size();
        nSrc = hgSrc->vhexen.size();

        field.resize(nDst);
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

    /* Dot product of each weight vector with the corresponding source sheet field values, multiplied by the strength of
     * the projection
     */
    void getWeightedSum(void) {
#pragma omp parallel for
        for (size_t i = 0; i < nDst; i++) {
            field[i] = 0.;
            for (size_t j = 0; j < counts[i]; j++) {
                field[i] += *fSrc[srcId[i][j]] * weights[i][j];
            }
            field[i] *= strength;
        }
    }

    /* Hebbian adaptation of the weights */
    void learn(void) {
        if (alpha > 0.0) {
#pragma omp parallel for
            for (size_t i = 0; i < nDst; i++) {
                for (size_t j = 0; j < counts[i]; j++) {
                    weights[i][j] += *fSrc[srcId[i][j]] * *fDst[i] * alphas[i];
                }
            }
        }
    }

    vector<Flt> getWeights(void) {
        vector<Flt> weightStore;
        for (size_t i = 0; i < nDst; i++) {
            for (size_t j = 0; j < counts[i]; j++) {
                weightStore.push_back(weights[i][j]);
            }
        }
        return weightStore;
    }

    void setWeights(vector<Flt> weightStore) {
        int k = 0;
        for (size_t i = 0; i < nDst; i++) {
            for (size_t j = 0; j < counts[i]; j++) {
                weights[i][j] = weightStore[k];
                k++;
            }
        }
    }

    void renormalize(void) {
#pragma omp parallel for
        for (size_t i = 0; i < nDst; i++) {
            Flt sumWeights = 0.0;
            for (size_t j = 0; j < counts[i]; j++) {
                sumWeights += weights[i][j];
            }
            for (size_t j = 0; j < counts[i]; j++) {
                weights[i][j] /= sumWeights;
            }
        }
    }

    void multiplyWeights(int i, double scale) {
#pragma omp parallel for
        for (size_t j = 0; j < counts[i]; j++) {
            weights[i][j] *= scale;
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
public:
    vector<Projection<Flt>> Projections;
    alignas(alignof(vector<Flt>)) vector<Flt> X;
    alignas(alignof(vector<Flt *>)) vector<Flt *> Xptr;
    vector<vector<int> > P;            // identity of projections to (potentially) joint normalize

    virtual void init(void) {
        this->stepCount = 0;
        this->zero_vector_variable(this->X);
    }

    virtual void allocate(void) {
        morph::RD_Base<Flt>::allocate();
        this->resize_vector_variable(this->X);
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            Xptr.push_back(&this->X[hi]);
        }
    }

    void addProjection(vector<Flt *> inXptr, HexGrid *hgSrc, float radius, float strength, float alpha, float sigma,
                       bool normalizeAlphas) {
        Projections.push_back(
                Projection<Flt>(inXptr, this->Xptr, hgSrc, this->hg, radius, strength, alpha, sigma, normalizeAlphas));
    }

    void zero_X(void) {
#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->X[hi] = 0.;
        }
    }

    void setNormalize(vector<int> proj) {
        for (int &p : proj) {
            for (size_t i = 0; i < this->P.size(); i++) {
                for (size_t j = 0; j < this->P[i].size(); j++) {
                    if (p == this->P[i][j]) {
                        cout << "Caution - projection may be mutiply joint normalized" << endl;
                    }
                }
            }
        }
        this->P.push_back(proj);
    }

    void renormalize(void) {
        for (size_t proj = 0; proj < this->P.size(); proj++) {
#pragma omp parallel for
            for (size_t i = 0; i < this->nhex; i++) {
                Flt sumWeights = 0.0;
                for (size_t p = 0; p < this->P[proj].size(); p++) {
                    for (size_t j = 0; j < this->Projections[this->P[proj][p]].counts[i]; j++) {
                        sumWeights += this->Projections[this->P[proj][p]].weights[i][j];
                    }
                }
                for (size_t p = 0; p < this->P[proj].size(); p++) {
                    for (size_t j = 0; j < this->Projections[this->P[proj][p]].counts[i]; j++) {
                        this->Projections[this->P[proj][p]].weights[i][j] /= sumWeights;
                    }
                }
            }
        }
    }
};

template<class Flt>
class LGN : public RD_Sheet<Flt> {
public:
    virtual void step(void) {
        this->stepCount++;
        this->zero_X();
        for (size_t i = 0; i < this->Projections.size(); i++) {
            this->Projections[i].getWeightedSum();
        }

        for (size_t i = 0; i < this->Projections.size(); i++) {
#pragma omp parallel for
            for (size_t hi = 0; hi < this->nhex; ++hi) {
                this->X[hi] += this->Projections[i].field[hi];
            }
        }
#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->X[hi] = fmax(this->X[hi], 0.);
        }
    }
};


template<class Flt>
class CortexSOM : public RD_Sheet<Flt> {
public:
    Flt beta, lambda, mu, oneMinusBeta, thetaInit;

    alignas(alignof(vector<Flt>)) vector<Flt> Xavg;
    alignas(alignof(vector<Flt>)) vector<Flt> Theta;

    virtual void init(void) {
        oneMinusBeta = (1. - beta);
        this->stepCount = 0;
        this->zero_vector_variable(this->X);
        this->zero_vector_variable(this->Xavg);
        this->zero_vector_variable(this->Theta);
    }

    virtual void allocate(void) {
        morph::RD_Base<Flt>::allocate();
        this->resize_vector_variable(this->X);
        this->resize_vector_variable(this->Xavg);
        this->resize_vector_variable(this->Theta);
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->Xptr.push_back(&this->X[hi]);
        }
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->Xavg[hi] = mu;
            this->Theta[hi] = thetaInit;
        }
    }

    virtual void step(void) {
        this->stepCount++;

        for (size_t i = 0; i < this->Projections.size(); i++) {
            this->Projections[i].getWeightedSum();
        }

        this->zero_X();

        for (size_t i = 0; i < this->Projections.size(); i++) {
#pragma omp parallel for
            for (size_t hi = 0; hi < this->nhex; ++hi) {
                this->X[hi] += this->Projections[i].field[hi];
            }
        }

#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->X[hi] = this->X[hi] - this->Theta[hi];
            if (this->X[hi] < 0.0) {
                this->X[hi] = 0.0;
            }
        }
    }

    virtual void step(vector<int> projectionIDs) {
        this->stepCount++;

        for (size_t i = 0; i < projectionIDs.size(); i++) {
            this->Projections[projectionIDs[i]].getWeightedSum();
        }

        this->zero_X();

        for (size_t i = 0; i < projectionIDs.size(); i++) {
#pragma omp parallel for
            for (size_t hi = 0; hi < this->nhex; ++hi) {
                this->X[hi] += this->Projections[projectionIDs[i]].field[hi];
            }
        }

#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->X[hi] = this->X[hi] - this->Theta[hi];
            if (this->X[hi] < 0.0) {
                this->X[hi] = 0.0;
            }
        }
    }

    void homeostasis(void) {
#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->Xavg[hi] = oneMinusBeta * this->X[hi] + beta * this->Xavg[hi];
        }
#pragma omp parallel for
        for (size_t hi = 0; hi < this->nhex; ++hi) {
            this->Theta[hi] = this->Theta[hi] + lambda * (this->Xavg[hi] - mu);
        }
    }
};

template<class Flt>
class PatternGenerator_Sheet : public RD_Sheet<Flt> {
public:
    virtual void step(void) {
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
            double xpos = ((double) i / (double) maxDim) - 0.5;
            for (int j = 0; j < ny; j++) {
                double ypos = ((double) j / (double) maxDim) - 0.5;
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

    virtual void step(void) {
        this->zero_X();
#pragma omp parallel for
        for (size_t i = 0; i < this->nhex; i++) {
            for (size_t j = 0; j < counts[i]; j++) {
                this->X[i] += C.vsquare[srcId[i][j]].X * weights[i][j];
            }
            this->X[i] *= strength;
        }
    }

    void stepCamera(void) {
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

    void stepPreloaded(void) {
        int p = floor(morph::Tools::randDouble() * PreLoadedPatterns.size());
        stepPreloaded(p);
    }
};