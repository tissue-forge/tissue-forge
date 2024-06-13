/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2023-2024 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

#include "tfColorMaps.h"

#include "colormaps/colormaps.h"

using namespace TissueForge;

struct ColormapItem {
    const char* name;
    rendering::ColorMapperFunc func;
};

#define COLORMAP_FUNCTION(CMAP) \
static fVector4 CMAP (rendering::ColorMapper *cm, const float& s) { \
    return fVector4{colormaps::all:: CMAP (s), 1};                  \
}


COLORMAP_FUNCTION(grey_0_100_c0);
COLORMAP_FUNCTION(grey_10_95_c0);
COLORMAP_FUNCTION(kryw_0_100_c71);
COLORMAP_FUNCTION(kryw_0_97_c73);
COLORMAP_FUNCTION(green_5_95_c69);
COLORMAP_FUNCTION(blue_5_95_c73);
COLORMAP_FUNCTION(bmw_5_95_c86);
COLORMAP_FUNCTION(bmy_10_95_c71);
COLORMAP_FUNCTION(bgyw_15_100_c67);
COLORMAP_FUNCTION(gow_60_85_c27);
COLORMAP_FUNCTION(gow_65_90_c35);
COLORMAP_FUNCTION(blue_95_50_c20);
COLORMAP_FUNCTION(red_0_50_c52);
COLORMAP_FUNCTION(green_0_46_c42);
COLORMAP_FUNCTION(blue_0_44_c57);
COLORMAP_FUNCTION(bwr_40_95_c42);
COLORMAP_FUNCTION(gwv_55_95_c39);
COLORMAP_FUNCTION(gwr_55_95_c38);
COLORMAP_FUNCTION(bkr_55_10_c35);
COLORMAP_FUNCTION(bky_60_10_c30);
COLORMAP_FUNCTION(bjy_30_90_c45);
COLORMAP_FUNCTION(bjr_30_55_c53);
COLORMAP_FUNCTION(bwr_55_98_c37);
COLORMAP_FUNCTION(cwm_80_100_c22);
COLORMAP_FUNCTION(bgymr_45_85_c67);
COLORMAP_FUNCTION(bgyrm_35_85_c69);
COLORMAP_FUNCTION(bgyr_35_85_c72);
COLORMAP_FUNCTION(mrybm_35_75_c68);
COLORMAP_FUNCTION(mygbm_30_95_c78);
COLORMAP_FUNCTION(wrwbw_40_90_c42);
COLORMAP_FUNCTION(grey_15_85_c0);
COLORMAP_FUNCTION(cgo_70_c39);
COLORMAP_FUNCTION(cgo_80_c38);
COLORMAP_FUNCTION(cm_70_c39);
COLORMAP_FUNCTION(cjo_70_c25);
COLORMAP_FUNCTION(cjm_75_c23);
COLORMAP_FUNCTION(kbjyw_5_95_c25);
COLORMAP_FUNCTION(kbw_5_98_c40);
COLORMAP_FUNCTION(bwy_60_95_c32);
COLORMAP_FUNCTION(bwyk_16_96_c31);
COLORMAP_FUNCTION(wywb_55_96_c33);
COLORMAP_FUNCTION(krjcw_5_98_c46);
COLORMAP_FUNCTION(krjcw_5_95_c24);
COLORMAP_FUNCTION(cwr_75_98_c20);
COLORMAP_FUNCTION(cwrk_40_100_c20);
COLORMAP_FUNCTION(wrwc_70_100_c20);

ColormapItem colormap_items[] = {
    {"Gray", grey_0_100_c0},
    {"DarkGray", grey_10_95_c0},
    {"Heat", kryw_0_100_c71},
    {"DarkHeat", kryw_0_97_c73},
    {"Green", green_5_95_c69},
    {"Blue", blue_5_95_c73},
    {"BlueMagentaWhite", bmw_5_95_c86},
    {"BlueMagentaYellow", bmy_10_95_c71},
    {"BGYW", bgyw_15_100_c67},
    {"GreenOrangeWhite", gow_60_85_c27},
    {"DarkGreenOrangeWhite", gow_65_90_c35},
    {"LightBlue", blue_95_50_c20},
    {"Red", red_0_50_c52},
    {"DarkGreen", green_0_46_c42},
    {"DarkBlue", blue_0_44_c57},
    {"BlueWhiteRed", bwr_40_95_c42},
    {"GreenWhiteViolet", gwv_55_95_c39},
    {"GreenWhiteRed", gwr_55_95_c38},
    {"BlueBlackRed", bkr_55_10_c35},
    {"BlueBlackYellow", bky_60_10_c30},
    {"BlueGrayYellow", bjy_30_90_c45},
    {"BlueGrayRed", bjr_30_55_c53},
    {"BluwWhiteRed", bwr_55_98_c37},
    {"CyanWhiteMagenta", cwm_80_100_c22},
    {"BGYMR", bgymr_45_85_c67},
    {"DarkBGYMR", bgyrm_35_85_c69},
    {"Rainbow", bgyr_35_85_c72},
    {"CyclicMRYBM", mrybm_35_75_c68},
    {"CyclicMYGBM", mygbm_30_95_c78},
    {"CyclicWRWBW", wrwbw_40_90_c42},
    {"CyclicGray", grey_15_85_c0},
    {"DarkCyanGreenOrange", cgo_70_c39},
    {"CyanGreenOrange", cgo_80_c38},
    {"CyanMagenta", cm_70_c39},
    {"CyanGrayOrange", cjo_70_c25},
    {"CyanGrayMagenta", cjm_75_c23},
    {"KBJW", kbjyw_5_95_c25},
    {"BlackBlueWhite", kbw_5_98_c40},
    {"BlueWhiteYellow", bwy_60_95_c32},
    {"CyclicBWYK", bwyk_16_96_c31},
    {"CyclicWYWB", wywb_55_96_c33},
    {"KRJCW", krjcw_5_98_c46},
    {"DarkKRJCW", krjcw_5_95_c24},
    {"CyclicCyanWhiteRed", cwr_75_98_c20},
    {"CyclicCWRK", cwrk_40_100_c20},
    {"CyclicWRWC", wrwc_70_100_c20}
};

static bool iequals(const std::string& a, const std::string& b)
{
    unsigned int sz = a.size();
    if (b.size() != sz)
        return false;
    for (unsigned int i = 0; i < sz; ++i)
        if (tolower(a[i]) != tolower(b[i]))
            return false;
    return true;
}

static int colormap_index_of_name(const char* s) {
    const int size = sizeof(colormap_items) / sizeof(ColormapItem);
    
    for(int i = 0; i < size; ++i) {
        if (iequals(s, colormap_items[i].name)) {
            return i;
        }
    }
    return -1;
}

rendering::ColorMapperFunc rendering::getColorMapperFunc(const std::string& name) {
    const int idx = colormap_index_of_name(name.c_str());
    if(idx < 0) return NULL;
    return colormap_items[idx].func;
}

std::vector<std::string> rendering::getColorMapperFuncNames() {
    std::vector<std::string> result;
    for(auto& i : colormap_items) 
        result.push_back(i.name);
    return result;
}
