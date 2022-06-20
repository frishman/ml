
def preprocess_nd(df):

    n_samples_before = len(df)
    df = df[df['Dx.Final'].notna()]
    df = df[df['Group'].notna()]
    n_samples_after = len(df)
    print("Excluded ", n_samples_before - n_samples_after, " rows out of ", n_samples_before)

    df["Age"].replace({"90+": "90"}, inplace=True)
    df["Age"] = df["Age"].astype('float64')

    df["Dx.Final"] = df["Dx.Final"].str.replace(' ', '_', regex=True)
    df["Dx.Final"] = df["Dx.Final"].str.replace(',', '_', regex=True)
    df["Dx.Final"] = df["Dx.Final"].str.replace('\'', '_', regex=True)
    df["Dx.Final"] = df["Dx.Final"].str.replace('\\/', '_', regex=True)
    df["Dx.Final"] = df["Dx.Final"].str.replace('(', '_', regex=True)
    df["Dx.Final"] = df["Dx.Final"].str.replace(')', '_', regex=True)

    df["Group"] = df["Group"].str.replace(' ', '_', regex=True)
    df["Group"] = df["Group"].str.replace('\'', '_', regex=True)
    df["Group"] = df["Group"].str.replace('(', '_', regex=True)
    df["Group"] = df["Group"].str.replace(')', '_', regex=True)

    return df
