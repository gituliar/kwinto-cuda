bench:
	./out/linux-release/src/kwinto bench test/portfolio.csv -v --put


release: config build check

build:
	cmake --build --preset linux-release
config:
	cmake --preset linux-release
check:
	./out/linux-release/test/kwinto_test --gtest_filter=-kwPortfolioTest.*
check-all:
	./out/linux-release/test/kwinto_test
