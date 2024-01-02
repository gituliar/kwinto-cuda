#include <iostream>
#include <vector>

#include "docopt.h"

#include "Core/kwConfig.h"
#include "Core/kwString.h"
#include "Pricer/kwPricerFactory.h"
#include "Utils/kwPortfolio.h"
#include "kwVersion.h"

using namespace kw;


constexpr char g_usage[] = R"(
kwinto - Options Pricing Analytics

Usage:
    kwinto price [options] <path>

Arguments:
    <path>          CSV file with options data

Options:
    -e <num>        Reject option prices less than <num> from RRMS error stats [default: 0.5]
    -n <num>        Run <num> batches (when 0 run all) [default: 0]
    -t <num>        Use <num> points for t-grid [default: 512]
    -x <num>        Use <num> points for x-grid [default: 512]
    -v              Show extra details, be verbose

    -h --help       Print this screen
    --version       Print Kwinto version
)";

using Args = std::map<std::string, docopt::value>;



kw::Error
    cmdPrice(const Args& args)
{
    kw::Config config;

    config.set("PRICER", "FD1D");

    config.set("FD1D.THETA", 0.5);
    config.set("FD1D.T_GRID_SIZE", args.at("-t").asLong());
    config.set("FD1D.X_GRID_SIZE", args.at("-x").asLong());

    double tolerance;
    if (auto error = kw::fromString(args.at("-e").asString(), tolerance); !error.empty())
        return "cmdBench: Fail to parse '-n <num>': " + error;

    /// Load Portfolio
    ///
    kw::Portfolio portfolio;
    if (auto err = portfolio.load(args.at("<path>").asString()); !err.empty())
        return "cmdPrice: " + err;

    std::cout << "Portfolio" << std::endl;
    std::cout << "    Assets : " << portfolio.assets().size() << std::endl;
    std::cout << std::endl;

    std::vector<f64> prices;
    if (auto err = portfolio.price(config, prices); !err.empty())
        return "cmdPrice: " + err;

    /// Check Prices
    ///
    if (auto err = portfolio.printPricesStats(prices); !err.empty())
        return "cmdPrice: " + err;

    return "";
}



int main(int argc, char** argv)
{
    auto version = "kwinto-cuda " + kw::Version::GIT_TAG + " (" + kw::Version::GIT_REV + ", " + kw::Version::BUILD_DATE + ")";


    auto args = docopt::docopt(
        g_usage,
        { argv + 1, argv + argc },
        true,
        version);
    std::cout << version << '\n' << std::endl;

    if (args.at("-v").asBool()) {
        std::cout << "Command-Line Arguments" << std::endl;
        for (auto const& arg : args) {
            std::cout << "    " << arg.first << ": " << arg.second << std::endl;
        }
        std::cout << std::endl;
    }


    kw::Error error;
    if (args["price"])
        error = cmdPrice(args);

    if (!error.empty()) {
        std::cerr << error << std::endl;
        return 1;
    }

    return 0;
}

