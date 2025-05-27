# run.sh
#!/bin/bash
docker build -t fairness-agent .
docker run --rm -v $(pwd)/results:/app/results fairness-agent
