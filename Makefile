bench:
	out/linux-release/src/kwinto price -p FD1D-BS --density 0.30 --scale 50 test/portfolio_qdfp.csv
	out/linux-release/src/kwinto price -p FD1D-BS --density 0.35 --scale 50 test/portfolio_qdfp.csv
	out/linux-release/src/kwinto price -p FD1D-BS --density 0.40 --scale 50 test/portfolio_qdfp.csv
	#out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 8192 -x 512 -t 512 | tee log/`hostname`_512_8192.log


release: config build check

build:
	cmake --build --preset linux-release
config:
	cmake --preset linux-release
check:
	./out/linux-release/test/kwinto_test --gtest_filter=-kwPortfolioTest.*
check-all:
	zstdcat test/portfolio_qdfp.csv.zst > test/portfolio_qdfp.csv
	./out/linux-release/test/kwinto_test
