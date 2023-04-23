release: config build check


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
	./bin/kwinto_test
