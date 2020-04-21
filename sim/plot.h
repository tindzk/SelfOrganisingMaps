#ifndef GCAL_PLOT_H
#define GCAL_PLOT_H

#include <morph/HexGrid.h>
#include <morph/RD_Plot.h>

void scalarfields (morph::RD_Plot<double>& plt,
                   Gdisplay& disp,
                   HexGrid* hg,
                   const vector<double*>& f,
                   double mina = +1e7, double maxa = -1e7, double overallOffset = 0.0);

#endif //GCAL_PLOT_H
