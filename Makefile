.PHONY: setup test test-python run-server run-desktop train convert build-firmware clean

setup:
	./scripts/setup_python.sh

test: test-python

test-python:
	python3 -m pytest tests/python -v

run-server:
	./scripts/run_server.sh

run-desktop:
	./scripts/run_desktop_demo.sh $(WAV)

train:
	./scripts/train_model.sh $(ARCH)

convert:
	./scripts/convert_model.sh $(ARCH)

build-firmware:
	./scripts/build_firmware.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache