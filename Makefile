release: config build check

run:
	./out/linux-release/src/kwinto bench test/portfolio.csv -v

build:
	cmake --build --preset linux-release
config:
	cmake --preset linux-release


install-kwinto:
	rm -f bin/kwinto
	upx -o bin/kwinto build/linux-release/src/kwinto
install-test:
	cp out/linux-release/test/kwinto_test bin/


check:
	./out/linux-release/test/kwinto_test --gtest_filter=-kwPortfolioTest.*
check-all:
	./out/linux-release/test/kwinto_test
