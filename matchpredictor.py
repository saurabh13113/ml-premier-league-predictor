import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Read match data from CSV file
matches = pd.read_csv("matches.csv", index_col=0)

# Drop unnecessary columns to clean up the dataset
del matches["comp"]
del matches["notes"]

# Convert 'date' to datetime format for easier manipulation
matches["date"] = pd.to_datetime(matches["date"])

# Encode categorical features to numerical codes
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

# Extract hour from time string and convert to integer
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)

# Extract day of the week from the date (Monday=0, Sunday=6)
matches["day_code"] = matches["date"].dt.dayofweek

# Create target variable: 1 for win ('W'), 0 otherwise
matches["target"] = (matches["result"] == "W").astype(int)

# Split data into training and test sets based on the date
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

# List of predictor columns to use for the model
predictors = ["venue_code", "opp_code", "hour", "day_code"]

# Initialize RandomForest model with specified parameters
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# Train the model on the training set
rf.fit(train[predictors], train["target"])

# Make predictions on the test set
preds = rf.predict(test[predictors])

# Calculate and print accuracy and precision scores
acc = accuracy_score(test["target"], preds)
print("Initial accuracy score:", acc)

# Combine actual and predicted results for analysis
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
print("Initial precision score:", precision_score(test["target"], preds))

# Function to calculate rolling averages for specified columns
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Columns to calculate rolling averages for
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages to each team
matches_rolling = matches.groupby("team").apply(
    lambda x: rolling_averages(x, cols, new_cols)
)
matches_rolling = matches_rolling.droplevel("team")
matches_rolling.index = range(matches_rolling.shape[0])

# Function to retrain model and make predictions on new data
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds),
                            index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

# Retrain model with rolling averages as additional predictors
combined, precision = make_predictions(matches_rolling, predictors + new_cols)
print("Second precision score:", precision)

# Combine home and away results for further analysis
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]],
                          left_index=True, right_index=True)

# Dictionary to handle team name variations
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester UTD",
    "Newcastle United": "Newcastle UTD",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)

# Merge predictions to analyze head-to-head matchups
merged = combined.merge(combined, left_on=["date", "new_team"],
                        right_on=["date", "opponent"])

# Print the count of actual wins where the model predicted a win for one team but not the other
print(merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)][
          "actual_x"].value_counts())

# Calculate final Premier League positions based on predicted results
# Calculate wins, losses, and draws
matches_rolling['predicted'] = rf.predict(matches_rolling[predictors + new_cols])
positions = matches_rolling.groupby("team").agg(
    wins=pd.NamedAgg(column="predicted", aggfunc="sum"),
    losses=pd.NamedAgg(column="predicted", aggfunc=lambda x: len(x) - x.sum()),
    draws=pd.NamedAgg(column="predicted", aggfunc=lambda x: len(x) - x.sum() - (len(x) - x.sum()))
).sort_values("wins", ascending=False)

# Print out final Premier League position table
positions["points"] = positions["wins"] * 3 + positions["draws"]
print("\nFinal Predicted Premier League Position Table:")
print(positions.sort_values("points", ascending=False))
