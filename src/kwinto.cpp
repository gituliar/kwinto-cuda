#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

#include "docopt.h"

#define KW_BENCHMARK_ON
#include "kwBenchmark.h"

#include "kwFd1d.h"
#include "kwTime.h"
#include "kwPortfolio.h"


namespace kw {

constexpr char usage[] = R"(
kwinto - Financial Analytics for Equity Market Modeling

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




template<typename Real, typename Pricer>
kw::Error
    benchPortfolio(const kw::Portfolio& portfolio, kw::Fd1dConfig config, size_t batchSize, size_t batchCount, double tolerance)
{
    Pricer pricer;

    if (batchSize == 0)
        batchSize = portfolio.size();
    config.pdeCount = batchSize;

    if (batchCount == 0)
        batchCount = (portfolio.size() + batchSize - 1) / batchSize;

    if (auto error = pricer.allocate(config); !error.empty())
        return "benchPortfolio: " + error;

    std::vector<std::vector<kw::Option>> batches(1);
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
            return "benchPortfolio: " + error;

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
            return "benchPortfolio: " + error;
        if (assets.size() == batchSize) { KW_BENCHMARK_PAUSE(label); }

        // 3. Check
        for (auto j = 0; j < assets.size(); ++j)
        {
            const auto& asset = assets[j];

            const auto& price = portfolio.at(asset);
            const auto spot = 100.;

            Real got;
            if (auto error = pricer.value(j, spot, got); !error.empty())
            {
                std::cerr << error << std::endl;
                continue;
            }
            if (std::abs(price - got) > tolerance)
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

    if (auto error = pricer.free(); !error.empty())
        return "benchPortfolio: " + error;

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

    double tolerance;
    if (auto error = kw::fromString(args.at("-e").asString(), tolerance); !error.empty())
        return "cmdBench: Fail to parse '-n <num>': " + error;


    kw::Portfolio portfolio;
    if (auto error = kw::loadPortfolio(args.at("<portfolio>").asString(), portfolio); !error.empty())
        return "cmdBench: " + error;

    std::cout << "Portfolio" << std::endl;
    std::cout << "    batch count: " << batchCount << std::endl;
    std::cout << "    batch size:  " << batchSize << std::endl;
    std::cout << "    total:       " << portfolio.size() << std::endl;
    std::cout << std::endl;


    if (auto error = benchPortfolio<double, kw::Fd1d<double>>(portfolio, config, batchSize, batchCount, tolerance); !error.empty())
        return "cmdBench: " + error;
    KW_BENCHMARK_PRINT("Fd1d<double>::solve");

    if (auto error = benchPortfolio<double, kw::Fd1d_Gpu<double>>(portfolio, config, batchSize, batchCount, tolerance); !error.empty())
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

