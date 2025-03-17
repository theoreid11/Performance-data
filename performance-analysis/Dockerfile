FROM python:3.11.5

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/outputs/shap_plots

# Set environment variables
ENV DATA_PATH="/app/data/dungeon_performance_ranks_28.01-24.02.csv"
ENV MODEL_DIR="/app/models"
ENV MODEL_FILE="model.joblib"
ENV METADATA_FILE="metadata.json"
ENV OUTPUT_DIR="/app/outputs/shap_plots"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy script
COPY performance_rank_analysis.py .

# Copy dataset
COPY data/dungeon_performance_ranks_28.01-24.02.csv /app/data/

# Run the script
CMD ["python", "performance_rank_analysis.py"]