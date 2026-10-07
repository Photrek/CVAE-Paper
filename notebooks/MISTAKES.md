# notebooks/ — mistakes

- **Shell scripts saved with Windows line endings.** On Colab, `./run_pipeline.sh` fails with `\r` errors. The notebook strips them with `sed -i 's/\r$//'` before running. Keep that step for any new script it launches.
