#include "kwFd1dQl.h"

#include <ql/qldefines.hpp>
#include <ql/exercise.hpp>
#include <ql/pricingengines/blackscholescalculator.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesvanillaengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>

#include <exception>

using namespace std::string_literals;


namespace ql = QuantLib;

namespace kw
{

Error
Fd1dQl_PriceEngine::init(const Config& config)
{
    m_tDim = config.get("FD1D.T_GRID_SIZE", 512);
    m_xDim = config.get("FD1D.X_GRID_SIZE", 512);

    return "";
}


Error
Fd1dQl_PriceEngine::price(const std::vector<Option>& assets, std::vector<double>& prices)
{
    auto n = assets.size();
    prices.resize(n);

    for (auto i = 0; i < n; i++) {
        if (auto error = priceOne(assets[i], prices[i]); !error.empty())
            return "Fd1dQl_PriceEngine::price: " + error;
    }

    return "";
}


Error
Fd1dQl_PriceEngine::priceOne(const Option& asset, double& price) const
{
    auto parity = (asset.w == Parity::Put) ? ql::Option::Put : ql::Option::Call;

    ql::ext::shared_ptr<ql::StrikedTypePayoff> payoff(new ql::PlainVanillaPayoff(parity, asset.k));

    // set up dates
    auto calendar = ql::TARGET();
    auto anchor = ql::Date(31, ql::Jul, 1944);
    auto settlementDate = anchor;

    ql::Settings::instance().evaluationDate() = anchor;

    auto dayCounter = ql::Actual365Fixed();

    auto maturity = anchor + std::ceil(DaysInYear * asset.t);

    ql::ext::shared_ptr<ql::Exercise> americanExercise(
        new ql::AmericanExercise(settlementDate, maturity));

    ql::Handle<ql::YieldTermStructure> flatTermStructure(
        ql::ext::shared_ptr<ql::YieldTermStructure>(
            new ql::FlatForward(settlementDate, asset.r, dayCounter)));
    ql::Handle<ql::YieldTermStructure> flatDividendTS(
        ql::ext::shared_ptr<ql::YieldTermStructure>(
            new ql::FlatForward(settlementDate, asset.q, dayCounter)));

    ql::VanillaOption americanOption(payoff, americanExercise);

    ql::Handle<ql::Quote> underlyingH(
        ql::ext::shared_ptr<ql::Quote>(new ql::SimpleQuote(asset.s)));

    ql::Handle<ql::BlackVolTermStructure> flatVolTS(
        ql::ext::shared_ptr<ql::BlackVolTermStructure>(
            new ql::BlackConstantVol(settlementDate, calendar, asset.z, dayCounter)));

    ql::ext::shared_ptr<ql::BlackScholesMertonProcess> bsmProcess(
        new ql::BlackScholesMertonProcess(underlyingH, flatDividendTS,
            flatTermStructure, flatVolTS));


    // FD engine for American Option
    auto engine = ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(
        bsmProcess, m_tDim, m_xDim, 0, ql::FdmSchemeDesc::CrankNicolson());
    americanOption.setPricingEngine(engine);

    try {
        price = americanOption.NPV();
    }
    catch (...) {
        std::exception_ptr ep = std::current_exception();
        try {
            std::rethrow_exception(ep);
        }
        catch (std::exception& e) {
            return "PriceEngine::calculateAmerican : "s + e.what();
        }
    }


    return "";
}

}