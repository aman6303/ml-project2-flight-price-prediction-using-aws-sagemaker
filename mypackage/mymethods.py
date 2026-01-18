# function to check if a destination city is north or not
import numpy as np
import pandas as pd


def is_north(X):
    columns = X.columns.to_list()

    north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
    return X.assign(
        **{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        }
    ).drop(columns=columns)


# method to check part of the day based on time


def part_of_day(X, morning=4, noon=12, eve=16, night=20):
    columns = X.columns.to_list()

    X_temp = X.assign(**{col: pd.to_datetime(X.loc[:, col]).dt.hour for col in columns})

    return X_temp.assign(
        **{
            f"{col}_part_of_the_day": np.select(
                [
                    X_temp.loc[:, col].between(morning, noon, inclusive="left"),
                    X_temp.loc[:, col].between(noon, eve, inclusive="left"),
                    X_temp.loc[:, col].between(eve, night, inclusive="left"),
                ],
                ["morning", "afternoon", "evening"],
                default="night",
            )
            for col in columns
        }
    ).drop(columns=columns)


# function for creating categories for duration


def duration_category(X, short=180, med=400):
    return X.assign(
        duration_cat=np.select(
            [X.duration.lt(short), X.duration.between(short, med, inclusive="left")],
            ["short", "medium"],
            default="long",
        )
    ).drop(columns="duration")


# function for checking id the duration is longer than given value


def is_over(X, value=1000):
    return X.assign(
        **{f"duration_over_{value}": X.duration.ge(value).astype(int)}
    ).drop(columns="duration")


# creating a new feature if its direct or not
def is_direct(X):
    return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))


# function for checing if there is info or not
def have_info(X):
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))
