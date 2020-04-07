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

    // HexGrid for LGN ON/OFF
    HexGrid* hgLgn;

    // LGN ON cells
    RD_Sheet<double> LGN_ON;

    // LGN OFF cells
    RD_Sheet<double> LGN_OFF;

    CortexSOM<double> CX;
    HexGrid* hgCx;

    // preferred orientation [0, π] for each cortical unit
    vector<double> preferredOrientation;

    // average orientation selectivity [0, 1] for each cortical unit
    vector<double> selectivity;

    // maximum selectivity across all simulations
    double maxSelectivity = lowestDouble;

    bool homeostasis;

    // Number of settling steps
    size_t settle;

    float beta, lambda, mu, thetaInit, xRange, yRange, afferAlpha, excitAlpha, inhibAlpha;
    float afferStrength, excitStrength, inhibStrength, LGNstrength, scale;

    float k, gamma_S, sigma_S;

    float sigmaA, sigmaB, afferRadius, excitRadius, inhibRadius, afferSigma, excitSigma, inhibSigma, LGNCenterSigma, LGNSurroundSigma;

    double* gainControlWeights;

    static constexpr double lowestDouble = std::numeric_limits<double>::lowest();

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
        afferStrength = root.get("afferStrength", 1.5).asFloat();  // gamma_A
        excitStrength = root.get("excitStrength", 1.7).asFloat();  // gamma_E
        inhibStrength = root.get("inhibStrength", -1.4).asFloat(); // gamma_I
        LGNstrength = root.get("LGNstrength", 14.0).asFloat();     // gamma_O

        // contrast-gain control for ON/OFF cells
        // parameters for spatial normalisation pooling S
        k       = 0.11;  // constant offset such that output is well-defined for weak inputs
        gamma_S = 0.6;   // strength of inhibitory gain control projection
        sigma_S = 0.125; // size of spatial normalisation pooling

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

        hgHcm = createHexGrid(root.get("IN_svgpath", "boundaries/trialmod.svg").asString());
        hgIn = createHexGrid(root.get("IN_svgpath", "boundaries/trialmod.svg").asString());
        hgLgn = createHexGrid(root.get("LGN_svgpath", "boundaries/trialmod.svg").asString());
        hgCx = createHexGrid(root.get("CX_svgpath", "boundaries/trialmod.svg").asString());

        auto squaresIn = squaresFromHexGrid(hgIn);
        auto squaresLgn = squaresFromHexGrid(hgLgn);
        auto squaresCx = squaresFromHexGrid(hgCx);

        // Mapping between Hexagonal and Cartesian Sheet
        HCM.init(hgHcm->num());

        // Input sheet
        IN.init(hgIn->num());

        // Gain control for LGN ON/OFF
        gainControlWeights = createWeightsGainControl<double>(squaresIn, squaresLgn, afferRadius, sigma_S);

        // LGN ON cells
        LGN_ON.init(hgLgn->num());
        LGN_ON.connect({
            Projection<double>(IN.X, createConnectionField<double>(squaresIn, squaresLgn, afferRadius, LGNCenterSigma, false), +LGNstrength, k, gamma_S, 0.0, false),
            Projection<double>(IN.X, createConnectionField<double>(squaresIn, squaresLgn, afferRadius, LGNSurroundSigma, false), -LGNstrength, k, gamma_S, 0.0, false)
        });

        renormalise(LGN_ON, {0});
        renormalise(LGN_ON, {1});

        // LGN OFF cells
        LGN_OFF.init(hgLgn->num());
        LGN_OFF.connect({
            // OFF weights are negation of ON weights, thus change signs of strength
            Projection<double>(IN.X, createConnectionField<double>(squaresIn, squaresLgn, afferRadius, LGNCenterSigma, false), -LGNstrength, k, gamma_S, 0.0, false),
            Projection<double>(IN.X, createConnectionField<double>(squaresIn, squaresLgn, afferRadius, LGNSurroundSigma, false), +LGNstrength, k, gamma_S, 0.0, false)
        });

        renormalise(LGN_OFF, {0});
        renormalise(LGN_OFF, {1});

        // Cortex Sheet (V1)
        CX.init(hgCx->num(), {.beta = beta, .mu = mu, .lambda = lambda, .thetaInit = thetaInit});
        // k = 1, gamma_S = 0 because no contrast-gain control for V1
        CX.connect({
            // afferent projections from ON/OFF cells
            Projection<double>(LGN_ON.X, createConnectionField<double>(squaresLgn, squaresCx, afferRadius, afferSigma, true), afferStrength * 0.5, 1, 0, afferAlpha, true),
            Projection<double>(LGN_OFF.X, createConnectionField<double>(squaresLgn, squaresCx, afferRadius, afferSigma, true), afferStrength * 0.5, 1, 0, afferAlpha, true),
            // recurrent lateral excitatory/inhibitory projections from other V1 cells
            Projection<double>(CX.X, createConnectionField<double>(squaresCx, squaresCx, excitRadius, excitSigma, false), excitStrength, 1, 0, excitAlpha, true),
            Projection<double>(CX.X, createConnectionField<double>(squaresCx, squaresCx, inhibRadius, inhibSigma, true), inhibStrength, 1, 0, inhibAlpha, true)
        });

        renormalise(CX, {0 /* LGN ON */, 1 /* LGN OFF */});
        renormalise(CX, {2 /* recurrent excitatory projection */});
        renormalise(CX, {3 /* recurrent inhibitory projection */});

        preferredOrientation.resize(CX.nhex, 0.);
        selectivity.resize(CX.nhex, 0.);
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
        sheetStep(LGN_ON, gainControlWeights);
        sheetStep(LGN_OFF, gainControlWeights);
    }

    void plotAfferent(morph::Gdisplay dispIn, morph::Gdisplay dispLgn) {
        vector<double> fx(3, 0.);
        RD_Plot<double> plt(fx, fx, fx);

        auto a = activations(IN);
        plt.scalarfields(dispIn, hgIn, a, 0., 1.0);

        vector<vector<double>> L = { activations(LGN_ON), activations(LGN_OFF) };
        plt.scalarfields(dispLgn, hgLgn, L);
    }

    /**
     * Cortical step
     *
     * @param f called for every settling step with grid and activations
     */
    void stepCortex(const std::function<void(HexGrid*, vector<double>&)> f) {
        zero_X(CX);  // Required because of CX's self connections

        // From paper: "Once all 16 settling steps are complete, the settled V1 activation pattern is deemed to be the
        // V1 response to the presented pattern."
        for (size_t j = 0; j < settle; j++) {
            sheetStep(CX, (double*) NULL);

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

        int i = 0;
        for (auto h : hgCx->hexen) {
            array<float, 3> cl = morph::Tools::HSVtoRGB(preferredOrientation[i] / M_PI, 1.0, selectivity[i]);
            disp.drawHex(h.position(), array<float, 3>{0., 0., 0.}, (h.d * 0.5f), cl);
            i++;
        }
        disp.redrawDisplay();
    }

    /**
     * Calculate preference map according to weighted average method described in Miikkulainen (2004), appendix G and
     * Stevens et al. (2013).
     *
     * Present shifted orientations [0, π] to the network as sine gratings. For each unit, calculate the
     * maximum phase from the cortical activations. Finally, compute the preferred orientation and average
     * selectivity for each unit.
     */
    void map() {
        // From Stevens et al. (2013, p. 8)
        size_t numOrientations = 20;
        size_t numPhases = 8;

        // The amplitude corresponds to the contrast.
        // Stevens et al. present gratings for a range of contrasts, but a fixed value can be used as long as
        // contrast-gain control is enabled.
        auto amplitude = 1.0;

        size_t gratingWidth = 30;

        // Sum of orientation preference vectors V
        vector<double> sumVx(CX.nhex);
        vector<double> sumVy(CX.nhex);

        // current orientation's maximum phase for each cortical unit
        vector<double> maxPhaseTemp(CX.nhex, lowestDouble);

        // maximum phase for each cortical unit across all orientations
        // corresponds to eta^_phi in G.1.3
        vector<double> maxPhase(CX.nhex, lowestDouble);

        for (size_t i = 0; i < numOrientations; i++) {
            std::fill(maxPhaseTemp.begin(), maxPhaseTemp.end(), lowestDouble);

            // current test orientation
            // orientations are π-periodic
            double theta = (double) i / numOrientations * M_PI;

            for (size_t j = 0; j < numPhases; j++) {
                // Grating() takes a phase value in the range [0, 2π]
                double phase = (double) j / numPhases * 2 * M_PI;

                // Present sine gratings
                IN.Grating(hgIn, theta, phase, gratingWidth, amplitude);

                // Perform LGN and CX sheet steps, but without lateral interactions
                // Note that this triggers the V1 activation function which could be omitted as per p. 8
                sheetStep(LGN_ON, gainControlWeights);
                sheetStep(LGN_OFF, gainControlWeights);
                // Do not perform any settling steps because "responses to afferent stimulation alone [...] provide a
                // sufficient approximation" (p. 478).
                sheetStep(
                        CX,
                        { CX.Projections[0] /* LGN ON */, CX.Projections[1] /* LGN OFF */ },
                        (double*) NULL);

                // Record peak responses for current orientation
                for (size_t k = 0; k < CX.nhex; k++)
                    maxPhaseTemp[k] = max(maxPhaseTemp[k], CX.X[k]);
            }

            for (size_t k = 0; k < CX.nhex; k++) {
                // Calculate orientation preference vector "with peak response as the length and 2θ as its orientation"
                // (Stevens et al., 2003).
                // Use factor 2 because activations are calculated for π-periodic orientations.
                sumVx[k] += maxPhaseTemp[k] * cos(2. * theta);
                sumVy[k] += maxPhaseTemp[k] * sin(2. * theta);
            }

            for (size_t k = 0; k < CX.nhex; k++)
                maxPhase[k] = max(maxPhase[k], maxPhaseTemp[k]);
        }

        for (size_t i = 0; i < CX.nhex; i++) {
            // Preferred orientation of unit i (eqn. G.2)
            // atan2(x, y) returns a value in range [-π, π]
            // `+ π` was added to G.2 such that the preferred orientation becomes positive, i.e. [0, π]
            preferredOrientation[i] = .5 * (atan2(sumVy[i], sumVx[i]) + M_PI);

            // Magnitude of summed vectors V (eqn. G.3)
            selectivity[i] = sqrt(sumVx[i] * sumVx[i] + sumVy[i] * sumVy[i]);
        }

        // Stevens et al. (2003, p. 7) use the maximum selectivity rather than the sum of selectivity values as in
        // Miikkulainen (2004, p. 477).
        // The maximum selectivity is calculated across all simulations (p. 7).
        for (size_t j = 0; j < CX.nhex; j++) maxSelectivity = max(maxSelectivity, selectivity[j]);

        // Calculate average orientation selectivity values
        // Eqn. G.3 (cont.)
        // The maximum selectivity is chosen instead of the sum of selectivity values such that selectivity[i] is in the
        // range [0, 1].
        for (size_t i = 0; i < CX.nhex; i++) selectivity[i] /= maxSelectivity;
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
