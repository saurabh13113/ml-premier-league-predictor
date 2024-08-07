# ml-premier-league-predictor-⚽
Summer Project that uses SciKit Learn and pandas to analyze data of the Premier League obtained from web-scraping, to predict match results for the season

## Premier League Match Prediction using Random Forest

This project uses a Random Forest Classifier to predict the outcomes of Premier League football matches. The model is trained on match data, including variables such as the venue, opponent, and match time, and is further improved by incorporating rolling averages of key match statistics.

## Project Structure

- **matches.csv**: The dataset containing match data.
- **predict_premier_league.py**: The main Python script containing the code for data processing, model training, and prediction.

## Setup Instructions

### Prerequisites

- Python 3.x
- Pandas
- Scikit-learn

### Running the Code

1. Clone the repository.
2. Ensure that `matches.csv` is in the same directory as the script.
3. Install the required libraries:
   ```bash
   pip install pandas scikit-learn

