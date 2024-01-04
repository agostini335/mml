def check_overlap(train_df, test_df, val_df):
    train_set = set(train_df)
    test_set = set(test_df)
    val_set = set(val_df)

    # Check for intersection between sets
    if train_set.intersection(test_set) or train_set.intersection(val_set) or test_set.intersection(val_set):
        overlapping_splits = train_set.intersection(test_set, val_set)
        print(f"Overlapping splits found: {overlapping_splits}")
        return False
    else:
        print("No overlapping splits found.")
        return True


def run_test(train_df=None, test_df=None, val_df=None):
    if check_overlap(train_df, test_df, val_df):
        print("Test passed.")
    else:
        raise ValueError("Test failed.")
