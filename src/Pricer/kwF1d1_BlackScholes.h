#pragma once

#include "Pricer/kwBlackScholes.h"
#include "Pricer/kwFd1d.h"


namespace kw {

class Fd1d_BlackScholes_Pricer : public Pricer
{
private:
    BlackScholes_Pricer
        m_bs;
    Fd1d_Pricer
        m_fd1d;

public:
    Error
        init(const Config& config) final;
    Error
        price(const vector<Option>& assets, vector<f64>& prices) final;
};
}