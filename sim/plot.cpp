#include <morph/tools.h>

#include "plot.h"

// Copied from Morphologica, adapted to take vector<double*> instead of vector<vector<double>>

/*!
 * On Gdisplay disp, plot all of the scalar fields stored in f on
 * the HexGrid hg. These are plotted in a row; it's up to the
 * programmer to make the window large enough when instantiating
 * the Gdisplay.
 *
 * Optionally pass in a min and a max to help scale the gradients
 *
 * @overallOffset can be optonally set to shift the fields in the horizontal
 * axis.
 */
void scalarfields (morph::RD_Plot<double>& plt,
                   Gdisplay& disp,
                   HexGrid* hg,
                   const vector<double*>& f,
                   double mina, double maxa, double overallOffset) {
    disp.resetDisplay (plt.fix, plt.eye, plt.rot);

    unsigned int N = f.size();
    unsigned int nhex = hg->num();

    // Determines min and max
    for (unsigned int hi=0; hi<nhex; ++hi) {
        Hex* h = hg->vhexen[hi];
        if (!h->onBoundary()) {
            for (unsigned int i = 0; i<N; ++i) {
                if (f[i][h->vi]>maxa) { maxa = f[i][h->vi]; }
                if (f[i][h->vi]<mina) { mina = f[i][h->vi]; }
            }
        }
    }

    double scalea = 1.0 / (maxa-mina);

    // Determine a colour from min, max and current value
    vector<vector<double> > norm_a;
    norm_a.resize (N);
    for (unsigned int i=0; i<N; ++i) {
        norm_a[i].resize (nhex, 0.0);
    }
    for (unsigned int i = 0; i<N; ++i) {
        for (unsigned int h=0; h<nhex; h++) {
            norm_a[i][h] = fmin (fmax (((f[i][h]) - mina) * scalea, 0.0), 1.0);
        }
    }

    // Create an offset which we'll increment by the width of the
    // map, starting from the left-most map (f[0])

    float hgwidth = hg->getXmax() - hg->getXmin();

    // Need to correctly apply N/2 depending on whether N is even or odd.
    float w = hgwidth+(hgwidth/20.0f);
    array<float,3> offset = { 0.0f , 0.0f, 0.0f };
    float half_minus_half_N = 0.5f - ((float)N/2.0f) + overallOffset;
    for (unsigned int i = 0; i<N; ++i) {
        offset[0] = (half_minus_half_N + (float)i) * w;
        // Note: OpenGL isn't thread-safe, so no omp parallel for here.
        for (auto h : hg->hexen) {

            // Colour can be single colour or red through to blue.
            array<float,3> cl_a = {{0,0,0}};
            if (plt.scalarFieldsSingleColour) {
                if (plt.singleColourHue >= 0.0 && plt.singleColourHue <= 1.0) {
                    cl_a = morph::Tools::HSVtoRGB (plt.singleColourHue, norm_a[i][h.vi], 1.0);
                } else {
                    cl_a = morph::Tools::HSVtoRGB ((float)i/(float)N, norm_a[i][h.vi], 1.0);
                }
            } else {
                cl_a = morph::Tools::getJetColorF (norm_a[i][h.vi]);
            }
            disp.drawHex (h.position(), offset, (h.d/2.0f), cl_a);
        }
    }
    disp.redrawDisplay();
}
