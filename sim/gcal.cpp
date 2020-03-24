#include <opencv2/opencv.hpp>

#include <morph/display.h>
#include <morph/tools.h>
#include <morph/HexGrid.h>
#include <morph/ReadCurves.h>
#include <morph/RD_Base.h>
#include <morph/RD_Plot.h>

#include <absl/strings/str_format.h>

using absl::PrintF;
using absl::FPrintF;
using absl::StrFormat;

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "topo.h"

using namespace morph;

class gcal : public Network {
public:
    HexCartSampler<double> HCM;
    HexGrid* hgHcm;

    PatternGenerator_Sheet<double> IN;
    HexGrid* hgIn;

    // LGN ON cells
    RD_Sheet<double> LGN_ON;
    HexGrid* hgLgnOn;

    // LGN OFF cells
    RD_Sheet<double> LGN_OFF;
    HexGrid* hgLgnOff;

    CortexSOM<double> CX;
    HexGrid* hgCx;

    vector<double> pref, sel;
    bool homeostasis;

    // Number of settling steps
    size_t settle;

    float beta, lambda, mu, thetaInit, xRange, yRange, afferAlpha, excitAlpha, inhibAlpha;
    float afferStrength, excitStrength, inhibStrength, LGNstrength, scale;
    float sigmaA, sigmaB, afferRadius, excitRadius, inhibRadius, afferSigma, excitSigma, inhibSigma, LGNCenterSigma, LGNSurroundSigma;

    void init(Json::Value root) {
        // Read parameters from JSON
        settle = root.get("settle", 16).asUInt();

        // homeostasis
        homeostasis = root.get("homeostasis", true).asBool();
        beta = root.get("beta", 0.991).asFloat();
        lambda = root.get("lambda", 0.01).asFloat();
        mu = root.get("thetaInit", 0.15).asFloat();
        thetaInit = root.get("mu", 0.024).asFloat();

        // Gaussian
        xRange = root.get("xRange", 2.0).asFloat();
        yRange = root.get("yRange", 2.0).asFloat();

        // learning rates
        afferAlpha = root.get("afferAlpha", 0.1).asFloat();
        excitAlpha = root.get("excitAlpha", 0.0).asFloat();
        inhibAlpha = root.get("inhibAlpha", 0.3).asFloat();

        // projection strengths
        afferStrength = root.get("afferStrength", 1.5).asFloat();
        excitStrength = root.get("excitStrength", 1.7).asFloat();
        inhibStrength = root.get("inhibStrength", -1.4).asFloat();
        LGNstrength = root.get("LGNstrength", 14.0).asFloat();

        // spatial params
        scale = root.get("scale", 0.5).asFloat();
        sigmaA = root.get("sigmaA", 1.0).asFloat() * scale;
        sigmaB = root.get("sigmaB", 0.3).asFloat() * scale;
        afferRadius = root.get("afferRadius", 0.27).asFloat() * scale;
        excitRadius = root.get("excitRadius", 0.1).asFloat() * scale;
        inhibRadius = root.get("inhibRadius", 0.23).asFloat() * scale;
        afferSigma = root.get("afferSigma", 0.270).asFloat() * scale;
        excitSigma = root.get("excitSigma", 0.025).asFloat() * scale;
        inhibSigma = root.get("inhibSigma", 0.075).asFloat() * scale;
        LGNCenterSigma = root.get("LGNCenterSigma", 0.037).asFloat() * scale;
        LGNSurroundSigma = root.get("LGNSuroundSigma", 0.150).asFloat() * scale;

        // INITIALIZE LOGFILE
        string logpath = root.get("logpath", "logs/").asString();
        morph::Tools::createDir(logpath);
        HdfData data(StrFormat("%s/log.h5", logpath));

        // Mapping between Hexagonal and Cartesian Sheet
        hgHcm = createHexGrid(root.get("IN_svgpath", "boundaries/trialmod.svg").asString());
        HCM.init(hgHcm->num());

        // Input sheet
        hgIn = createHexGrid(root.get("IN_svgpath", "boundaries/trialmod.svg").asString());
        IN.init(hgIn->num());

        // LGN ON cells
        hgLgnOn = createHexGrid(root.get("LGN_svgpath", "boundaries/trialmod.svg").asString());
        LGN_ON.init(hgLgnOn->num());
        LGN_ON.connect({
            Projection<double>(IN.X, createConnections<double>(squaresFromHexGrid(hgIn), squaresFromHexGrid(hgLgnOn), afferRadius, LGNCenterSigma), +LGNstrength, 0.0, false),
            Projection<double>(IN.X, createConnections<double>(squaresFromHexGrid(hgIn), squaresFromHexGrid(hgLgnOn), afferRadius, LGNSurroundSigma), -LGNstrength, 0.0, false)
        });

        renormalise(LGN_ON, {0});
        renormalise(LGN_ON, {1});

        // LGN OFF cells
        hgLgnOff = createHexGrid(root.get("IN_svgpath", "boundaries/trialmod.svg").asString());
        LGN_OFF.init(hgLgnOff->num());
        LGN_OFF.connect({
            Projection<double>(IN.X, createConnections<double>(squaresFromHexGrid(hgIn), squaresFromHexGrid(hgLgnOff), afferRadius, LGNCenterSigma), -LGNstrength, 0.0, false),
            Projection<double>(IN.X, createConnections<double>(squaresFromHexGrid(hgIn), squaresFromHexGrid(hgLgnOff), afferRadius, LGNSurroundSigma), +LGNstrength, 0.0, false)
        });

        renormalise(LGN_OFF, {0});
        renormalise(LGN_OFF, {1});

        // Cortex Sheet (V1)
        hgCx = createHexGrid(root.get("CX_svgpath", "boundaries/trialmod.svg").asString());
        CX.init(hgCx->num(), {.beta = beta, .mu = mu, .lambda = lambda, .thetaInit = thetaInit});
        CX.connect({
            // afferent projection from ON/OFF cells
            Projection<double>(LGN_ON.X, createConnections<double>(squaresFromHexGrid(hgLgnOn), squaresFromHexGrid(hgCx), afferRadius, afferSigma), afferStrength, afferAlpha, true),
            Projection<double>(LGN_OFF.X, createConnections<double>(squaresFromHexGrid(hgLgnOff), squaresFromHexGrid(hgCx), afferRadius, afferSigma), afferStrength, afferAlpha, true),
            // recurrent lateral excitatory/inhibitory projection from other V1 cells
            Projection<double>(CX.X, createConnections<double>(squaresFromHexGrid(hgCx), squaresFromHexGrid(hgCx), excitRadius, excitSigma), excitStrength, excitAlpha, true),
            Projection<double>(CX.X, createConnections<double>(squaresFromHexGrid(hgCx), squaresFromHexGrid(hgCx), inhibRadius, inhibSigma), inhibStrength, inhibAlpha, true)
        });

        renormalise(CX, {0 /* LGN ON */, 1 /* LGN OFF */});
        renormalise(CX, {2 /* recurrent excitatory projection */});
        renormalise(CX, {3 /* recurrent inhibitory projection */});

        pref.resize(CX.nhex, 0.);
        sel.resize(CX.nhex, 0.);
    }

    /**
     * Performs an afferent step, i.e. present input to LGN ON/OFF
     * @note Does not perform step on cortical sheet
     */
    void stepAfferent(unsigned type) {
        switch (type) {
            case 0: { // Gaussians
                IN.Gaussian(
                        hgIn,
                        (morph::Tools::randDouble() - 0.5) * xRange,
                        (morph::Tools::randDouble() - 0.5) * yRange,
                        morph::Tools::randDouble() * M_PI, sigmaA, sigmaB);
            }
                break;
            case 1: { // Preloaded
                HCM.stepPreloaded();
                copyActivations(HCM, IN);
            }
                break;
            case 2: { // Camera input
                HCM.stepCamera();
                copyActivations(HCM, IN);
            }
                break;
            default: {
                for (int i = 0; i < HCM.C.n; i++) {
                    HCM.C.vsquare[i].X = morph::Tools::randDouble();
                }
                HCM.step();
                copyActivations(HCM, IN);
            }
        }
        sheetStep(LGN_ON);
        sheetStep(LGN_OFF);
    }

    void plotAfferent(morph::Gdisplay dispIn, morph::Gdisplay dispLgn) {
        vector<double> fx(3, 0.);
        RD_Plot<double> plt(fx, fx, fx);

        auto a = activations(IN);
        plt.scalarfields(dispIn, hgIn, a, 0., 1.0);

        vector<vector<double>> L = { activations(LGN_ON), activations(LGN_OFF) };
        plt.scalarfields(dispLgn, hgLgnOn, L);
    }

    /**
     * Cortical step
     *
     * @param f called for every settling step with grid and activations
     */
    void stepCortex(const std::function<void(HexGrid*, vector<double>&)> f) {
        zero_X(CX);

        // From paper: "Once all 16 settling steps are complete, the settled V1 activation pattern is deemed to be the
        // V1 response to the presented pattern."
        for (size_t j = 0; j < settle; j++) {
            sheetStep(CX);

            auto a = activations(CX);
            f(hgCx, a);
        }

        // From paper: "V1 afferent connection weights [...] from the ON/OFF sheets are adjusted once per iteration
        // (after V1 settling is completed) using a simple Hebbian learning rule."
        // "Weights are normalized separately for each of the other projections, to ensure that Hebbian learning does
        // not disrupt the balance between feedforward drive, lateral and feedback inhibition."
        learn(CX, {0, 1}); /* LGN ON and OFF */
        learn(CX, {2}); /* recurrent excitatory projection */
        learn(CX, {3}); /* recurrent inhibitory projection */

        if (homeostasis) CX.homeostasis();
        time++;
    }

    void plotWeights(morph::Gdisplay disp, int id) {
        vector<double> fix(3, 0.);
        RD_Plot<double> plt(fix, fix, fix);
        vector<vector<double> > W;
        W.push_back(CX.Projections[0].getWeightPlot(id));
        W.push_back(CX.Projections[1].getWeightPlot(id));
        W.push_back(CX.Projections[3].getWeightPlot(id));
        plt.scalarfields(disp, hgCx, W);
    }

    void plotMap(morph::Gdisplay disp) {
        disp.resetDisplay(vector<double>(3, 0.), vector<double>(3, 0.), vector<double>(3, 0.));
        double maxSel = -1e9;
        for (size_t i = 0; i < CX.nhex; i++) {
            if (sel[i] > maxSel) { maxSel = sel[i]; }
        }
        maxSel = 1. / maxSel;
        double overPi = 1. / M_PI;

        int i = 0;
        for (auto h : hgCx->hexen) {
            array<float, 3> cl = morph::Tools::HSVtoRGB(pref[i] * overPi, 1.0, sel[i] * maxSel);
            disp.drawHex(h.position(), array<float, 3>{0., 0., 0.}, (h.d * 0.5f), cl);
            i++;
        }
        disp.redrawDisplay();
    }

    void map() {
        size_t nOr = 20;
        size_t nPhase = 8;
        auto phaseInc = M_PI / (double) nPhase;
        vector<int> maxIndOr(CX.nhex, 0);
        vector<double> maxValOr(CX.nhex, -1e9);
        vector<double> maxPhase(CX.nhex, 0.);
        vector<double> Vx(CX.nhex);
        vector<double> Vy(CX.nhex);

        // Do not perform any steps for CX's self connections
        vector<Projection<double>> afferent = {
                CX.Projections[0] /* LGN ON */,
                CX.Projections[1] /* LGN OFF */
        };

        for (size_t i = 0; i < nOr; i++) {
            double theta = i * M_PI / (double) nOr;
            std::fill(maxPhase.begin(), maxPhase.end(), -1e9);
            for (size_t j = 0; j < nPhase; j++) {
                double phase = j * phaseInc;
                IN.Grating(hgIn, theta, phase, 30.0, 1.0);
                sheetStep(LGN_ON);
                sheetStep(LGN_OFF);
                zero_X(CX);  // Required because of CX's self connections
                sheetStep(CX, afferent);
                for (size_t k = 0; k < maxPhase.size(); k++) {
                    if (maxPhase[k] < CX.X[k]) maxPhase[k] = CX.X[k];
                }
            }

            for (size_t k = 0; k < maxPhase.size(); k++) {
                Vx[k] += maxPhase[k] * cos(2.0 * theta);
                Vy[k] += maxPhase[k] * sin(2.0 * theta);
            }

            for (size_t k = 0; k < maxPhase.size(); k++) {
                if (maxValOr[k] < maxPhase[k]) {
                    maxValOr[k] = maxPhase[k];
                    maxIndOr[k] = i;
                }
            }
        }

        for (size_t i = 0; i < maxValOr.size(); i++) {
            pref[i] = 0.5 * (atan2(Vy[i], Vx[i]) + M_PI);
            sel[i] = sqrt(Vy[i] * Vy[i] + Vx[i] * Vx[i]);
        }
    }

    void save(const string filename) {
        HdfData data(filename);
        vector<int> timetmp(1, time);
        data.add_contained_vals("time", timetmp);
        for (size_t p = 0; p < CX.Projections.size(); p++) {
            auto proj = CX.Projections[p].getWeights();
            data.add_contained_vals(StrFormat("proj_%i", p).c_str(), proj);
        }
    }

    void load(const string filename) {
        HdfData data(filename, 1);
        vector<int> timetmp;
        data.read_contained_vals("time", timetmp);
        time = timetmp[0];
        for (size_t p = 0; p < CX.Projections.size(); p++) {
            vector<double> proj;
            data.read_contained_vals(StrFormat("proj_%i", p).c_str(), proj);
            CX.Projections[p].setWeights(proj);
        }
        PrintF("Loaded weights and modified time to %i", time);
    }
};


/*** MAIN PROGRAM ***/

ABSL_FLAG(std::string, configFile, "configs/config.json", "Configuration file");
ABSL_FLAG(int, seed, 1, "Seed");
ABSL_FLAG(int, mode, 1, "Mode");
ABSL_FLAG(int, input, 0, "Input (0 = Gaussian, 1 = Loaded, 2 = Camera input)");
ABSL_FLAG(std::string, weightFile, "", "Weight file");
ABSL_FLAG(std::string, outputPath, "", "Output path for weights/images");

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    auto paramsfile = absl::GetFlag(FLAGS_configFile);
    auto seed = absl::GetFlag(FLAGS_seed);
    srand(seed);       // set seed
    auto MODE = absl::GetFlag(FLAGS_mode);
    auto INTYPE = absl::GetFlag(FLAGS_input);

    //  Set up JSON code for reading the parameters
    std::ifstream jsonfile_test;
    int srtn = system("pwd");
    if (srtn) FPrintF(stderr, "system call returned %s\n", srtn);

    jsonfile_test.open(paramsfile, std::ios::in);
    if (jsonfile_test.is_open()) jsonfile_test.close(); // Good, file exists.
    else {
        FPrintF(stderr, "JSON configuration file %s not found.\n", paramsfile);
        return 1;
    }

    // Parse the JSON
    std::ifstream jsonfile(paramsfile, std::ifstream::binary);
    Json::Value root;
    string errs;
    Json::CharReaderBuilder rbuilder;
    rbuilder["collectComments"] = false;
    bool parsingSuccessful = Json::parseFromStream(rbuilder, jsonfile, &root, &errs);
    if (!parsingSuccessful) {
        cerr << "Failed to parse JSON: " << errs;
        return 1;
    }

    size_t nBlocks = root.get("blocks", 100).asUInt();
    size_t steps = root.get("steps", 100).asUInt();

    // Creates the network
    gcal Net;
    Net.init(root);

    // Input specific setup
    switch (INTYPE) {
        case (0): { // Gaussian patterns
        }
            break;
        case (1): {   // preload patterns
            int ncols = root.get("cameraCols", 100).asUInt();
            int nrows = root.get("cameraRows", 100).asUInt();
            Net.HCM.initProjection(Net.hgHcm, ncols, nrows, 0.01, 20.);
            string filename = root.get("patterns", "configs/testPatterns.h5").asString();
            Net.HCM.preloadPatterns(filename);
        }
            break;

        case (2): {
            int ncols = root.get("cameraCols", 100).asUInt();
            int nrows = root.get("cameraRows", 100).asUInt();
            int stepsize = root.get("cameraSampleStep", 7).asUInt();
            int xoff = root.get("cameraOffsetX", 100).asUInt();
            int yoff = root.get("cameraOffsetY", 0).asUInt();
            Net.HCM.initProjection(Net.hgHcm, ncols, nrows, 0.01, 20.);
            if (!Net.HCM.initCamera(xoff, yoff, stepsize)) { return 0; }
        }
            break;
    }

    auto weightFile = absl::GetFlag(FLAGS_weightFile);
    if (weightFile.empty()) PrintF("Using random weights");
    else {
        PrintF("Using weight file: %s", weightFile);
        Net.load(weightFile);
    }

    auto outputPath = absl::GetFlag(FLAGS_outputPath);

    switch (MODE) {
        case 0: { // No plotting
            auto plotActivations = [](HexGrid*, vector<double>&) {};
            for (size_t b = 0; b < nBlocks; b++) {
                Net.map();
                for (size_t i = 0; i < steps; i++) {
                    Net.stepAfferent(INTYPE);
                    Net.stepCortex(plotActivations);
                }
                if (!outputPath.empty())
                    Net.save(StrFormat("%s/weights_%i.h5", outputPath, Net.time));
            }
        }
            break;

        case 1: { // Plotting
            vector<morph::Gdisplay> displays;
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Input Activity", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Cortical Activity", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(1200, 400, 0, 0, "Cortical Projection", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 300, 0, 0, "LGN ON/OFF", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Map", 1.7, 0.0, 0.0));

            vector<double> fx(3, 0.);
            RD_Plot<double> plt(fx, fx, fx);
            auto plotActivations = [&displays, fx, &plt](HexGrid* hg, vector<double>& X) {
                plt.scalarfields(displays[1], hg, X);
            };

            for (auto &d: displays) {
                d.resetDisplay(vector<double>(3, 0), vector<double>(3, 0), vector<double>(3, 0));
                d.redrawDisplay();
            }
            for (size_t b = 0; b < nBlocks; b++) {
                Net.map();
                Net.plotMap(displays[4]);
                for (size_t i = 0; i < steps; i++) {
                    Net.stepAfferent(INTYPE);
                    Net.plotAfferent(displays[0], displays[3]);
                    Net.stepCortex(plotActivations);
                    Net.plotWeights(displays[2], 500);
                }
                if (!outputPath.empty())
                    Net.save(StrFormat("%s/weights_%i.h5", outputPath, Net.time));
            }
            for (auto &d: displays) d.closeDisplay();
        }
            break;

        case 2: { // Map only
            vector<morph::Gdisplay> displays;
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Input Activity", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Cortical Activity", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(1200, 400, 0, 0, "Cortical Projection", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 300, 0, 0, "LGN ON/OFF", 1.7, 0.0, 0.0));
            displays.push_back(morph::Gdisplay(600, 600, 0, 0, "Map", 1.7, 0.0, 0.0));

            vector<double> fx(3, 0.);
            RD_Plot<double> plt(fx, fx, fx);
            auto plotActivations = [&displays, fx, &plt](HexGrid* hg, vector<double>& X) {
                plt.scalarfields(displays[1], hg, X);
            };

            for (auto &d: displays) {
                d.resetDisplay(vector<double>(3, 0), vector<double>(3, 0), vector<double>(3, 0));
                d.redrawDisplay();
            }
            Net.map();
            Net.plotMap(displays[4]);
            Net.stepAfferent(INTYPE);
            Net.plotAfferent(displays[0], displays[3]);
            Net.stepCortex(plotActivations);
            Net.plotWeights(displays[2], 500);
            for (size_t i = 0; i < displays.size(); i++) {
                displays[i].redrawDisplay();
                if (!outputPath.empty())
                    displays[i].saveImage(StrFormat("%s/plot_%i_%i.png", outputPath, Net.time, i));
            }
            for (auto &d: displays) d.closeDisplay();
        }
            break;
    }

    return 0;
}
