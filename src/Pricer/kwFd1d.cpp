#include "Pricer/kwFd1d.h"

#include <algorithm>
#include <tuple>

using namespace kw;


Error
Fd1d_Pricer::init(const Config& config)
{
    m_tDim = config.get("FD1D.T_GRID_SIZE", 512);
    m_xDim = config.get("FD1D.X_GRID_SIZE", 512);

    return "";
}

Error
Fd1d_Pricer::price(const vector<Option>& assets, vector<f64>& prices)
{
    const auto n = assets.size();
    if (n == 0)
        return "";

    /// Compression (allows to solve one PDE per options chain)
    ///
    vector<u32> asset2pde;
    vector<u32> pde2asset;
    {
        auto assetLess = [](const Option& l, const Option& r) -> bool {
            return std::tie(l.t, l.r, l.q, l.z, l.e, l.w) < std::tie(r.t, r.r, r.q, r.z, r.e, r.w);
        };

        auto assetEqual = [&assetLess](const Option& l, const Option& r) -> bool {
            return assetLess(l, r) == assetLess(r, l);
        };

        vector<u32> assetsSorted;
        assetsSorted.reserve(assets.size());
        for (auto i = 0; i < assets.size(); i++)
            assetsSorted.push_back(i);

        std::sort(assetsSorted.begin(), assetsSorted.end(), [&assets, &assetLess](const u32& l, const u32& r) {
            return assetLess(assets[l], assets[r]);
        });


        asset2pde.resize(assets.size());
        asset2pde[assetsSorted[0]] = 0;
        pde2asset.push_back(assetsSorted[0]);
        for (auto i = 1, k = 0; i < assetsSorted.size(); i++) {
            auto l = assetsSorted[i-1];
            auto r = assetsSorted[i];

            if (!assetEqual(assets[l], assets[r])) {
                k += 1;
                pde2asset.push_back(r);
            }

            asset2pde[r] = k;
        }
    }


    vector<Fd1dPde> pdes;
    const auto m = pde2asset.size();

    /// Fill PDE coefficients
    ///
    pdes.reserve(m);
    for (const auto& i : pde2asset) {
        auto& pde = pdes.emplace_back();

        const auto& asset = assets[i];

        pde.t = asset.t;

        pde.a0 = -asset.r;
        pde.ax = asset.r - asset.q - asset.z * asset.z / 2;
        pde.axx = asset.z * asset.z / 2;

        pde.earlyExercise = asset.e;
    }


    /// Allocate working memory
    ///
    m_t.resize(m, m_tDim);
    m_x.resize(m, m_xDim);
    m_v.resize(m, m_xDim);

    /// Fill t-grid
    ///
    for (auto i = 0; i < m; ++i) {
        f64 tMin = 0.0, tMax = pdes[i].t;
        f64 dt = (tMax - tMin) / (m_tDim - 1);

        for (auto j = 0; j < m_tDim; ++j)
            m_t(i, j) = tMin + j * dt;
    }

    /// Fill x-grid
    /// 
    /// https://github.com/lballabio/QuantLib/blob/master/ql/methods/finitedifferences/meshers/fdmblackscholesmesher.cpp
    /// https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/fdblackscholesvanillaengine.cpp
    /// 
    for (auto i = 0; i < m; ++i) {
        const auto& asset = assets[pde2asset[i]];

        const f64 density = 0.5;
        const f64 scale = 50;

        const f64 xMid = 0.; // log(asset.s / asset.k);
        const f64 xMin = xMid - scale * asset.z * sqrt(asset.t);
        const f64 xMax = xMid + scale * asset.z * sqrt(asset.t);

        const f64 yMin = asinh((xMin - xMid) / density);
        const f64 yMax = asinh((xMax - xMid) / density);

        const f64 dy = 1. / (m_xDim - 1);
        for (auto j = 0; j < m_xDim; ++j) {
            const f64 yj = j * dy;
            m_x(i, j) = xMid + density * sinh(yMin * (1.0 - yj) + yMax * yj);
        }
    }

    /// Fill v-grid
    ///
    for (auto i = 0; i < m; ++i) {
        const auto& asset = assets[pde2asset[i]];
        for (auto j = 0; j < m_xDim; ++j) {
            if (asset.w < 0)
                /// put
                m_v(i,j) = std::max<f64>(0, 1. - exp(m_x(i,j)));
            else
                /// call
                m_v(i,j) = std::max<f64>(0, exp(m_x(i,j)) - 1.);
        }
    }

    /// Solve
    ///
    if (auto error = m_solver.solve(pdes, m_t, m_x, m_v); !error.empty())
        return "Fd1d_Pricer::price: " + error;

    /// Fill prices
    ///
    prices.resize(n);
    for (auto i = 0; i < n; i++) {
        const auto& asset = assets[i];

        f64 price_;
        auto x = log(asset.s / asset.k);
        if (auto err = m_solver.value(asset2pde[i], x, m_x, price_); !err.empty())
            return "Fd1d_Pricer::price " + err;
        prices[i] = asset.k * price_;
    }

    return "";
}
