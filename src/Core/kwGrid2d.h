#pragma once

#include <vector>

#include "Core/kwTypes.h"


namespace kw {

class Grid2d {
private:
    std::vector<f64>
        m_buf;

    u64 m_nCol;
    u64 m_nRow;

public:
    Grid2d(const u64 nCol = 0, const u64 nRow = 0) :
        m_nCol{ nCol },
        m_nRow{ nRow }
    {
        m_buf.resize(nCol * nRow);
    };

    /// Getters
    ///
    u64 cols() const { return m_nCol; }

    u64 rows() const { return m_nRow; }

    const f64&
        operator()(const u64 col, const u64 row) const { return m_buf[col * m_nRow + row]; };

    /// Setters
    ///
    f64&
        operator()(const u64 col, const u64 row) { return m_buf[col * m_nRow + row]; };

    /// Allocator
    ///
    void
        resize(const u64 nCol, const u64 nRow)
    {
        m_buf.resize(nCol * nRow);

        m_nCol = nCol;
        m_nRow = nRow;
    }
};

}
