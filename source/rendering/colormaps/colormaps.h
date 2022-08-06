#pragma once

#include <array>
#include <algorithm>

#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>

namespace colormaps {
    typedef float real;
    typedef Magnum::Color3 rgb_color;

    namespace all {
        namespace {
            inline real lerp(int i, real x, const real *values, size_t N = 256) {
                x = std::max(std::min(x, 1.0f), 0.0f);
                real dx = x*N;
                int ix = std::min((int)std::trunc(x*(N - 1)), (int)(N - 2));
                real f_lower = values[ix * 3 + i];
                real f_upper = values[(ix + 1) * 3 + i];
                return f_lower + (dx - ix)*(f_upper - f_lower);
            }
        }

        inline rgb_color grey_0_100_c0(real x) {
            static const real values[] = {
#include "CETperceptual/linear_grey_0-100_c0_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color grey_10_95_c0(real x) {
            static const real values[] = {
#include "CETperceptual/linear_grey_10-95_c0_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color kryw_0_100_c71(real x) {
            static const real values[] = {
#include "CETperceptual/linear_kryw_0-100_c71_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color kryw_0_97_c73(real x) {
            static const real values[] = {
#include "CETperceptual/linear_kryw_0-97_c73_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color green_5_95_c69(real x) {
            static const real values[] = {
#include "CETperceptual/linear_green_5-95_c69_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color blue_5_95_c73(real x) {
            static const real values[] = {
#include "CETperceptual/linear_blue_5-95_c73_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bmw_5_95_c86(real x) {
            static const real values[] = {
#include "CETperceptual/linear_bmw_5-95_c86_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bmy_10_95_c71(real x) {
            static const real values[] = {
#include "CETperceptual/linear_bmy_10-95_c71_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bgyw_15_100_c67(real x) {
            static const real values[] = {
#include "CETperceptual/linear_bgyw_15-100_c67_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color gow_60_85_c27(real x) {
            static const real values[] = {
#include "CETperceptual/linear_gow_60-85_c27_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color gow_65_90_c35(real x) {
            static const real values[] = {
#include "CETperceptual/linear_gow_65-90_c35_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color blue_95_50_c20(real x) {
            static const real values[] = {
#include "CETperceptual/linear_blue_95-50_c20_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color red_0_50_c52(real x) {
            static const real values[] = {
#include "CETperceptual/linear_ternary-red_0-50_c52_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color green_0_46_c42(real x) {
            static const real values[] = {
#include "CETperceptual/linear_ternary-green_0-46_c42_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color blue_0_44_c57(real x) {
            static const real values[] = {
#include "CETperceptual/linear_ternary-blue_0-44_c57_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bwr_40_95_c42(real x) {
            static const real values[] = {
#include "CETperceptual/diverging_bwr_40-95_c42_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color gwv_55_95_c39(real x) {
            static const real values[] = {
#include "CETperceptual/diverging_gwv_55-95_c39_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color gwr_55_95_c38(real x) {
            static const real values[] = {
#include "CETperceptual/diverging_gwr_55-95_c38_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bkr_55_10_c35(real x) {
            static const real values[] = {
#include "CETperceptual/diverging_bkr_55-10_c35_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bky_60_10_c30(real x) {
            static const real values[] = {
#include "CETperceptual/diverging_bky_60-10_c30_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bjy_30_90_c45(real x) {
            static const real values[] = {
#include "CETperceptual/diverging-linear_bjy_30-90_c45_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bjr_30_55_c53(real x) {
            static const real values[] = {
#include "CETperceptual/diverging-linear_bjr_30-55_c53_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bwr_55_98_c37(real x) {
            static const real values[] = {
#include "CETperceptual/diverging_bwr_55-98_c37_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color cwm_80_100_c22(real x) {
            static const real values[] = {
#include "CETperceptual/diverging_cwm_80-100_c22_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bgymr_45_85_c67(real x) {
            static const real values[] = {
#include "CETperceptual/diverging-rainbow_bgymr_45-85_c67_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bgyrm_35_85_c69(real x) {
            static const real values[] = {
#include "CETperceptual/rainbow_bgyrm_35-85_c69_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bgyr_35_85_c72(real x) {
            static const real values[] = {
#include "CETperceptual/rainbow_bgyr_35-85_c72_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color mrybm_35_75_c68(real x) {
            static const real values[] = {
#include "CETperceptual/cyclic_mrybm_35-75_c68_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color mygbm_30_95_c78(real x) {
            static const real values[] = {
#include "CETperceptual/cyclic_mygbm_30-95_c78_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color wrwbw_40_90_c42(real x) {
            static const real values[] = {
#include "CETperceptual/cyclic_wrwbw_40-90_c42_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color grey_15_85_c0(real x) {
            static const real values[] = {
#include "CETperceptual/cyclic_grey_15-85_c0_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color cgo_70_c39(real x) {
            static const real values[] = {
#include "CETperceptual/isoluminant_cgo_70_c39_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color cgo_80_c38(real x) {
            static const real values[] = {
#include "CETperceptual/isoluminant_cgo_80_c38_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color cm_70_c39(real x) {
            static const real values[] = {
#include "CETperceptual/isoluminant_cm_70_c39_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color cjo_70_c25(real x) {
            static const real values[] = {
#include "CETperceptual/diverging-isoluminant_cjo_70_c25_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color cjm_75_c23(real x) {
            static const real values[] = {
#include "CETperceptual/diverging-isoluminant_cjm_75_c23_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color kbjyw_5_95_c25(real x) {
            static const real values[] = {
#include "CETperceptual/linear-protanopic-deuteranopic_kbjyw_5-95_c25_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color kbw_5_98_c40(real x) {
            static const real values[] = {
#include "CETperceptual/linear-protanopic-deuteranopic_kbw_5-98_c40_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bwy_60_95_c32(real x) {
            static const real values[] = {
#include "CETperceptual/diverging-protanopic-deuteranopic_bwy_60-95_c32_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color bwyk_16_96_c31(real x) {
            static const real values[] = {
#include "CETperceptual/cyclic-protanopic-deuteranopic_bwyk_16-96_c31_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color wywb_55_96_c33(real x) {
            static const real values[] = {
#include "CETperceptual/cyclic-protanopic-deuteranopic_wywb_55-96_c33_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color krjcw_5_98_c46(real x) {
            static const real values[] = {
#include "CETperceptual/linear-tritanopic_krjcw_5-98_c46_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color krjcw_5_95_c24(real x) {
            static const real values[] = {
#include "CETperceptual/linear-tritanopic_krjcw_5-95_c24_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color cwr_75_98_c20(real x) {
            static const real values[] = {
#include "CETperceptual/diverging-tritanopic_cwr_75-98_c20_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color cwrk_40_100_c20(real x) {
            static const real values[] = {
#include "CETperceptual/cyclic-tritanopic_cwrk_40-100_c20_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

        inline rgb_color wrwc_70_100_c20(real x) {
            static const real values[] = {
#include "CETperceptual/cyclic-tritanopic_wrwc_70-100_c20_n256.csv"
            };
            return{ lerp(0, x, values), lerp(1, x, values), lerp(2, x, values) };
        }

    }
    namespace linear {
        using all::grey_0_100_c0;
        using all::grey_10_95_c0;
        using all::kryw_0_100_c71;
        using all::kryw_0_97_c73;
        using all::green_5_95_c69;
        using all::blue_5_95_c73;
        using all::bmw_5_95_c86;
        using all::bmy_10_95_c71;
        using all::bgyw_15_100_c67;
        using all::gow_60_85_c27;
        using all::gow_65_90_c35;
        using all::blue_95_50_c20;
        namespace diverging {
            using all::bjy_30_90_c45;
            using all::bjr_30_55_c53;
        }
        namespace ternary {
            using all::red_0_50_c52;
            using all::green_0_46_c42;
            using all::blue_0_44_c57;
        }
        namespace colorblind {
            namespace protanopic_deuteranopic {
                using all::kbjyw_5_95_c25;
                using all::kbw_5_98_c40;
            }
            namespace tritanopic {
                using all::krjcw_5_98_c46;
                using all::krjcw_5_95_c24;
            }
            using namespace protanopic_deuteranopic;
            using namespace tritanopic;
        }
        using namespace diverging;
        using namespace ternary;
        using namespace colorblind;
    }
    namespace diverging {
        using all::bwr_40_95_c42;
        using all::gwv_55_95_c39;
        using all::gwr_55_95_c38;
        using all::bkr_55_10_c35;
        using all::bky_60_10_c30;
        using all::bwr_55_98_c37;
        using all::cwm_80_100_c22;
        namespace linear {
            using all::bjy_30_90_c45;
            using all::bjr_30_55_c53;
        }
        namespace rainbow {
            using all::bgymr_45_85_c67;
        }
        namespace isoluminant {
            using all::cjo_70_c25;
            using all::cjm_75_c23;
        }
        namespace colorblind {
            namespace protanopic_deuteranopic {
                using all::bwy_60_95_c32;
            }
            namespace tritanopic {
                using all::cwr_75_98_c20;
            }
            using namespace protanopic_deuteranopic;
            using namespace tritanopic;
        }
        using namespace linear;
        using namespace rainbow;
        using namespace isoluminant;
        using namespace colorblind;
    }
    namespace rainbow {
        using all::bgyrm_35_85_c69;
        using all::bgyr_35_85_c72;
        namespace diverging {
            using all::bgymr_45_85_c67;
        }
        using namespace diverging;
    }
    namespace cyclic {
        using all::mrybm_35_75_c68;
        using all::mygbm_30_95_c78;
        using all::wrwbw_40_90_c42;
        using all::grey_15_85_c0;
        namespace colorblind {
            namespace protanopic_deuteranopic {
                using all::bwyk_16_96_c31;
                using all::wywb_55_96_c33;
            }
            namespace tritanopic {
                using all::cwrk_40_100_c20;
                using all::wrwc_70_100_c20;
            }
            using namespace protanopic_deuteranopic;
            using namespace tritanopic;
        }
        using namespace colorblind;
    }
    namespace isoluminant {
        using all::cgo_70_c39;
        using all::cgo_80_c38;
        using all::cm_70_c39;
        namespace diverging {
            using all::cjo_70_c25;
            using all::cjm_75_c23;
        }
        using namespace diverging;
    }
    namespace colorblind {
        namespace protanopic_deuteranopic {
            namespace linear {
                using all::kbjyw_5_95_c25;
                using all::kbw_5_98_c40;
            }
            namespace diverging {
                using all::bwy_60_95_c32;
            }
            namespace cyclic {
                using all::bwyk_16_96_c31;
                using all::wywb_55_96_c33;
            }
            using namespace linear;
            using namespace diverging;
            using namespace cyclic;
        }
        namespace tritanopic {
            namespace linear {
                using all::krjcw_5_98_c46;
                using all::krjcw_5_95_c24;
            }
            namespace diverging {
                using all::cwr_75_98_c20;
            }
            namespace cyclic {
                using all::cwrk_40_100_c20;
                using all::wrwc_70_100_c20;
            }
            using namespace linear;
            using namespace diverging;
            using namespace cyclic;
        }
        using namespace protanopic_deuteranopic;
        using namespace tritanopic;
    }
    using namespace all;
}
