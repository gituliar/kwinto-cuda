bench:
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 128 | tee log/`hostname`_512_128.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 256 | tee log/`hostname`_512_256.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 512 | tee log/`hostname`_512_512.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 1024 | tee log/`hostname`_512_1024.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 2048 | tee log/`hostname`_512_2048.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 4096 | tee log/`hostname`_512_4096.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 8192 | tee log/`hostname`_512_8192.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 16384 | tee log/`hostname`_512_16384.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 32768 | tee log/`hostname`_512_32768.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 128 | tee log/`hostname`_1024_128.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 256 | tee log/`hostname`_1024_256.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 512 | tee log/`hostname`_1024_512.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 1024 | tee log/`hostname`_1024_1024.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 2048 | tee log/`hostname`_1024_2048.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 4096 | tee log/`hostname`_1024_4096.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 8192 | tee log/`hostname`_1024_8192.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 16384 | tee log/`hostname`_1024_16384.log
	out/linux-release/src/kwinto bench test/portfolio.csv -v --cpu32 --cpu64 --gpu32 --gpu64 -n 8 -b 32768 | tee log/`hostname`_1024_32768.log


release: config build check

build:
	cmake --build --preset linux-release
config:
	cmake --preset linux-release
check:
	./out/linux-release/test/kwinto_test --gtest_filter=-kwPortfolioTest.*
check-all:
	./out/linux-release/test/kwinto_test
