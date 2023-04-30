#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

#include "docopt.h"

#define KW_BENCHMARK_ON
#include "kwBenchmark.h"

#include "kwFd1d.h"
#include "kwTime.h"
#include "kwString.h"


namespace kw {

constexpr char usage[] = R"(
kwinto - Financial Analytics

Usage:
    kwinto bench [options] <portfolio>

Arguments:
    <portfolio>     Read options data from <portfolio> CSV file

Options:
    -b <num>        Price <num> options per batch [default: 128]
    --cuda-n <num>  Run <num> threads per CUDA block over the n-dimension [default: 1]
    --cuda-x <num>  Run <num> threads per CUDA block over the x-dimension [default: 128]
    -e <num>        Accept <num> error for calculated option prices [default: 0.01]
    -n <num>        Run <num> batches (when 0 run all) [default: 0]
    -o <file>       Write benchmark results into <file>
    -t <num>        Use <num> points for t-grid [default: 2048]
    -x <num>        Use <num> points for x-grid [default: 2048]
    -v              Show extra details, be verbose
    -h --help       Print this screen
    --version       Print Kwinto version

)";

}

using Args = std::map<std::string, docopt::value>;

template<typename Real>
using Portfolio = std::map<kw::Option<Real>, std::vector<std::pair<Real, Real>>>;



template<typename Real>
kw::Error
    loadPortfolio(const Args& args, Portfolio<Real>& portfolio)
{
    const auto srcPath = std::filesystem::absolute(args.at("<portfolio>").asString());
    std::ifstream src(srcPath);
    if (!src.is_open())
        return "loadPortfolio: Failed to open " + srcPath.string();

    int e, q, r, s, t, v, w, z;
    {
        std::string header;
        std::getline(src, header);

        int i = 0;
        e = q = r = s = t = v = w = z = -1;
        for (const auto& colName : kw::split(header, ','))
        {
            if (colName == "early_exercise")
                e = i;
            else if (colName == "dividend_rate")
                q = i;
            else if (colName == "interest_rate")
                r = i;
            else if (colName == "spot")
                s = i;
            else if (colName == "time_to_maturity")
                t = i;
            else if (colName == "price")
                v = i;
            else if (colName == "parity")
                w = i;
            else if (colName == "volatility")
                z = i;

            ++i;
        }
        if (e == -1 || q == -1 || r == -1 || s == -1 || t == -1 || v == -1 || w == -1 || z == -1)
        {
            std::stringstream error;
            error << "loadPortfolio: Some option data is missing: e=" << e << ", q=" << q << ", r=" << r
                << ", s=" << s << ", t=" << t << ", v=" << v << ", w=" << w << ", z=" << z;
            return error.str();
        }
    }

    for (std::string line; std::getline(src, line);)
    {
        auto vals = kw::split(line, ',');

        Real price, spot;
        kw::fromString(vals[v], price);
        kw::fromString(vals[s], spot);

        kw::Option<Real> asset;
        asset.k = 100;
        kw::fromString(vals[t], asset.t);
        kw::fromString(vals[z], asset.z);
        kw::fromString(vals[q], asset.q);
        kw::fromString(vals[r], asset.r);
        asset.e = (vals[e] == "a");
        asset.w = vals[w] == "c" ? kw::kParity::Call : kw::kParity::Put;

        portfolio[asset].emplace_back(spot, price);
    }

    return "";
}

template<typename Real, typename Pricer>
kw::Error
    pricePortfolio(const Portfolio<Real>& portfolio, kw::Fd1dConfig config, size_t batchSize, size_t batchCount)
{
    Pricer pricer;

    if (batchSize == 0)
        batchSize = portfolio.size();
    config.pdeCount = batchSize;

    if (batchCount == 0)
        batchCount = (portfolio.size() + batchSize - 1) / batchSize;

    if (auto error = pricer.allocate(config); !error.empty())
        return "pricePortfolio: " + error;

    std::vector<std::vector<kw::Option<Real>>> batches(1);
    {
        auto assets = std::ref(batches[0]);
        for (const auto& [asset, _] : portfolio)
        {
            if (assets.get().size() == batchSize)
                assets = batches.emplace_back();
            assets.get().push_back(asset);
        }
    }

    for (auto i = 0; i < batchCount; ++i)
    {
        const auto& assets = batches[i];

        std::vector<kw::Fd1dPde<Real>> pdes;

        // 1. Init
        if (auto error = kw::Fd1dPdeFor(assets, config, pdes); !error.empty())
            return "pricePortfolio: " + error;

        // 2. Solve
        std::string label;
        if constexpr (std::is_same_v<Pricer, kw::Fd1d<float>>)
            label = "Fd1d<float>::solve";
        else if constexpr (std::is_same_v<Pricer, kw::Fd1d<double>>)
            label = "Fd1d<double>::solve";
        else if constexpr (std::is_same_v<Pricer, kw::Fd1d_Gpu<float>>)
            label = "Fd1d_Gpu<float>::solve";
        else if constexpr (std::is_same_v<Pricer, kw::Fd1d_Gpu<double>>)
            label = "Fd1d_Gpu<double>::solve";
        else
            static_assert("unexpected Pricer type");

        if (assets.size() == batchSize) { KW_BENCHMARK_RESUME(label); }
        if (auto error = pricer.solve(pdes); !error.empty())
            return "pricePortfolio: " + error;
        if (assets.size() == batchSize) { KW_BENCHMARK_PAUSE(label); }

        // 3. Check
        for (auto j = 0; j < assets.size(); ++j)
        {
            const auto& asset = assets[j];

            for (const auto& [spot, price] : portfolio.at(asset))
            {
                Real got;
                if (auto error = pricer.value(j, spot, got); !error.empty())
                {
                    std::cerr << error << std::endl;
                    continue;
                }
                if (std::abs(price - got) > 0.01)
                {
                    std::cerr << "id:     " << i << ":" << j << std::endl;
                    std::cerr << "pricer: " << label << std::endl;
                    std::cerr << "want:   " << price << std::endl;
                    std::cerr << "got:    " << got << std::endl;
                    std::cerr << "diff:   " << price - got << std::endl;
                    std::cerr << "spot:   " << spot << std::endl;
                    std::cerr << "asset:  " << asset << std::endl;
                    std::cerr << std::endl;
                }
            }
        }
    }

    if (auto error = pricer.free(); !error.empty())
        return "pricePortfolio: " + error;

    return "";
}

kw::Error
    cmdBench(const Args& args)
{
    kw::Fd1dConfig config;
    config.theta = 0.5;
    config.tDim = args.at("-t").asLong();
    config.xDim = args.at("-x").asLong();
    config.xThreads = args.at("--cuda-x").asLong();
    config.nThreads = args.at("--cuda-n").asLong();

    auto batchCount = args.at("-n").asLong();
    auto batchSize = args.at("-b").asLong();


    Portfolio<double> portfolio;
    if (auto error = loadPortfolio(args, portfolio); !error.empty())
        return "cmdBench: " + error;

    std::cout << "Portfolio" << std::endl;
    std::cout << "    batch count: " << batchCount << std::endl;
    std::cout << "    batch size:  " << batchSize << std::endl;
    std::cout << "    total:       " << portfolio.size() << std::endl;
    std::cout << std::endl;


    if (auto error = pricePortfolio<double, kw::Fd1d<double>>(portfolio, config, batchSize, batchCount); !error.empty())
        return "cmdBench: " + error;
    KW_BENCHMARK_PRINT("Fd1d<double>::solve");

    if (auto error = pricePortfolio<double, kw::Fd1d_Gpu<double>>(portfolio, config, batchSize, batchCount); !error.empty())
        return "cmdBench: " + error;
    KW_BENCHMARK_PRINT("Fd1d_Gpu<double>::solve");

    return "";
}



int main(int argc, char** argv)
{
    auto args = docopt::docopt(
        kw::usage,
        { argv + 1, argv + argc },
        true,
        "kwinto v0.1.0");

    if (args.at("-v").asBool())
    {
        std::cout << "Command-Line Arguments" << std::endl;
        for (auto const& arg : args) {
            std::cout << "    " << arg.first << ": " << arg.second << std::endl;
        }
        std::cout << std::endl;
    }


    kw::Error error;
    if (args["bench"])
        error = cmdBench(args);

    if (!error.empty())
    {
        std::cerr << error << std::endl;
        return 1;
    }

    return 0;
}

