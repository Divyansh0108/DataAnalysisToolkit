# Core Pkgs
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import io

# Set up the UI aesthetics
st.set_page_config(page_title="🔍 Comprehensive Data Analysis Toolkit", layout="wide")
st.title("🔍 Comprehensive Data Analysis Toolkit 📊📈")
st.write(
    "Welcome to the Comprehensive Data Analysis Toolkit! This powerful tool enables you to perform Exploratory Data Analysis (EDA), Data Cleaning, Feature Engineering, Data Transformation, and more. With just a few clicks, you can visualize your data, clean and preprocess it, engineer features, and even build and evaluate machine learning models."
)


def download_link(df, filename):
    """Generate a link to download the dataset."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return st.download_button(
        label="Download Modified Dataset",
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
    )


def main():
    """Main function to run the Streamlit app"""
    activities = [
        "EDA",
        "Plots",
        "Scaling",
        "Data Cleaning",
        "Feature Engineering",
        "Data Transformation",
        "Model Building",
        "Hyperparameter Tuning",
        "Cross-Validation",
        "Data Sampling",
    ]
    choice = st.sidebar.selectbox("Select Activities", activities)

    if choice in activities:
        data = st.file_uploader(
            "Upload a Dataset", type=["csv", "txt", "xlsx", "xls", "pkl", "json"]
        )
        if data is not None:
            if data.name.endswith("csv"):
                df = pd.read_csv(data)
            elif data.name.endswith("txt"):
                df = pd.read_csv(data, delimiter="\t")
            elif data.name.endswith("xlsx") or data.name.endswith("xls"):
                df = pd.read_excel(data)
            elif data.name.endswith("pkl"):
                df = pd.read_pickle(data)
            elif data.name.endswith("json"):
                df = pd.read_json(data)

            all_columns = df.columns.to_list()

            st.dataframe(df.head(), use_container_width=True)

            if choice == "EDA":
                st.subheader("Exploratory Data Analysis")

                if st.checkbox("Show Shape"):
                    st.write(df.shape)

                if st.checkbox("Show Columns"):
                    st.write(all_columns)

                if st.checkbox("Summary"):
                    st.write(df.describe())

                if st.checkbox("Show Selected Columns"):
                    selected_columns = st.multiselect("Select Columns", all_columns)
                    if selected_columns:
                        new_df = df[selected_columns]
                        st.dataframe(new_df, use_container_width=True)

                if st.checkbox("Show Value Counts"):
                    if df.shape[1] > 0:
                        counts = df.iloc[:, -1].value_counts().reset_index()
                        counts.columns = ["Category", "Count"]
                        fig = px.bar(
                            counts, x="Category", y="Count", title="Value Counts"
                        )
                        st.plotly_chart(fig)

                if st.checkbox("Correlation Plot (Plotly Heatmap)"):
                    corr = df.corr()
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.columns,
                            colorscale="Viridis",
                        )
                    )
                    fig.update_layout(
                        title="Correlation Heatmap",
                        xaxis_title="Features",
                        yaxis_title="Features",
                    )
                    st.plotly_chart(fig)

                if st.checkbox("Pie Chart"):
                    if df.shape[1] > 0:
                        fig = px.pie(
                            df,
                            names=df.columns[0],
                            title="Pie Chart Distribution",
                            hole=0.3,
                            template="plotly_dark",
                        )
                        fig.update_traces(
                            textinfo="percent+label", pull=[0.1] * len(df)
                        )
                        st.plotly_chart(fig)

                download_link(df, "eda_dataset.csv")

            elif choice == "Plots":
                st.subheader("Data Visualization")

                if st.checkbox("Show Value Counts"):
                    if df.shape[1] > 0:
                        counts = df.iloc[:, -1].value_counts().reset_index()
                        counts.columns = ["Category", "Count"]
                        fig = px.bar(
                            counts, x="Category", y="Count", title="Value Counts"
                        )
                        st.plotly_chart(fig)

                # Customizable Plot
                plot_types = [
                    "Area Chart",
                    "Bar Chart",
                    "Line Chart",
                    "Histogram",
                    "Box Plot",
                    "Density Contour",
                    "Scatter Plot",
                    "Violin Plot",
                    "Pie Chart",
                ]
                type_of_plot = st.selectbox("Select Type of Plot", plot_types)
                selected_columns_names = st.multiselect(
                    "Select Columns To Plot", all_columns
                )

                if st.button("Generate Plot"):
                    st.success(
                        f"Generating Customizable Plot of {type_of_plot} for {selected_columns_names}"
                    )

                    if selected_columns_names:
                        cust_data = df[selected_columns_names]

                        if type_of_plot == "Area Chart":
                            st.line_chart(
                                cust_data
                            )  # Streamlit area chart is not available, use line chart instead

                        elif type_of_plot == "Bar Chart":
                            st.bar_chart(cust_data)

                        elif type_of_plot == "Line Chart":
                            st.line_chart(cust_data)

                        elif type_of_plot == "Histogram":
                            fig = px.histogram(cust_data)
                            st.plotly_chart(fig)

                        elif type_of_plot == "Box Plot":
                            fig = px.box(cust_data)
                            st.plotly_chart(fig)

                        elif type_of_plot == "Density Contour":
                            fig = px.density_contour(cust_data)
                            st.plotly_chart(fig)

                        elif type_of_plot == "Scatter Plot":
                            if len(selected_columns_names) == 2:
                                fig = px.scatter(
                                    df,
                                    x=selected_columns_names[0],
                                    y=selected_columns_names[1],
                                    title="Scatter Plot",
                                )
                                st.plotly_chart(fig)
                            else:
                                st.warning("Scatter Plot requires exactly two columns")

                        elif type_of_plot == "Violin Plot":
                            if len(selected_columns_names) == 2:
                                fig = px.violin(
                                    df,
                                    y=selected_columns_names[0],
                                    x=selected_columns_names[1],
                                    title="Violin Plot",
                                )
                                st.plotly_chart(fig)
                            else:
                                st.warning("Violin Plot requires exactly two columns")

                        elif type_of_plot == "Pie Chart":
                            if len(selected_columns_names) == 1:
                                fig = px.pie(
                                    df,
                                    names=selected_columns_names[0],
                                    title="Pie Chart Distribution",
                                    hole=0.3,
                                    template="plotly_dark",
                                )
                                fig.update_traces(
                                    textposition="outside",
                                    textinfo="percent+label",
                                    pull=[0.1],
                                )
                                fig.update_layout(showlegend=True)
                                st.plotly_chart(fig)
                            else:
                                st.warning("Pie Chart requires exactly one column")
                    else:
                        st.warning("Please select at least one column for plotting")

                download_link(df, "plots_dataset.csv")

            elif choice == "Scaling":
                st.subheader("Scaling Methods")

                # Make sure `all_columns` is defined here
                all_columns = df.columns.to_list()

                scaling_method = st.selectbox(
                    "Select Scaling Method",
                    ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                )
                columns_to_scale = st.multiselect(
                    "Select Columns to Scale", all_columns
                )

                if st.button("Apply Scaling"):
                    if columns_to_scale:
                        if scaling_method == "StandardScaler":
                            scaler = StandardScaler()
                        elif scaling_method == "MinMaxScaler":
                            scaler = MinMaxScaler()
                        elif scaling_method == "RobustScaler":
                            scaler = RobustScaler()

                        df[columns_to_scale] = scaler.fit_transform(
                            df[columns_to_scale]
                        )
                        st.success(f"Scaling applied using {scaling_method}")
                        st.dataframe(df.head(), use_container_width=True)
                    else:
                        st.warning("Please select the columns to scale!")

                download_link(df, "scaled_dataset.csv")

            elif choice == "Data Cleaning":
                st.subheader("Data Cleaning")

                if st.checkbox("Show Missing Values"):
                    st.write(df.isnull().sum())

                if st.checkbox("Drop Missing Values"):
                    df = df.dropna()
                    st.success("Missing values dropped")
                    st.dataframe(df.head(), use_container_width=True)

                if st.checkbox("Fill Missing Values"):
                    fill_value = st.text_input(
                        "Enter fill value (e.g., 'mean', 'median', '0')", "mean"
                    )
                    if fill_value in ["mean", "median"]:
                        df = df.fillna(
                            df.mean() if fill_value == "mean" else df.median()
                        )
                    else:
                        df = df.fillna(fill_value)
                    st.success("Missing values filled")
                    st.dataframe(df.head(), use_container_width=True)

                if st.checkbox("Remove Duplicates"):
                    df = df.drop_duplicates()
                    st.success("Duplicates removed")
                    st.dataframe(df.head(), use_container_width=True)

                if st.checkbox("Convert Data Types"):
                    column_name = st.selectbox("Select Column to Convert", all_columns)
                    dtype = st.selectbox("Select Data Type", ["int", "float", "str"])
                    if column_name and dtype:
                        df[column_name] = df[column_name].astype(dtype)
                        st.success(f"Column '{column_name}' converted to {dtype}")
                        st.dataframe(df.head(), use_container_width=True)

                download_link(df, "cleaned_dataset.csv")

            elif choice == "Feature Engineering":
                st.subheader("Feature Engineering")

                if st.checkbox("Create Polynomial Features"):
                    degree = st.slider("Select Degree", 2, 5, 2)
                    numeric_features = df.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    if numeric_features:
                        poly = PolynomialFeatures(degree)
                        df_poly = pd.DataFrame(
                            poly.fit_transform(df[numeric_features]),
                            columns=poly.get_feature_names_out(numeric_features),
                        )
                        df_poly = df_poly.add_suffix(
                            "_poly"
                        )  # Avoid column name conflicts
                        df = df.join(df_poly)
                        st.success(f"Polynomial features added with degree {degree}")
                        st.dataframe(df.head(), use_container_width=True)
                    else:
                        st.warning(
                            "No numeric columns available for polynomial feature creation."
                        )

                if st.checkbox("Binning"):
                    col = st.selectbox(
                        "Select Column for Binning",
                        df.select_dtypes(include=[np.number]).columns.tolist(),
                    )
                    bins = st.slider("Number of Bins", 2, 20, 5)
                    if col:
                        df[f"{col}_binned"] = pd.cut(df[col], bins=bins)
                        st.success(f"Binning applied to {col} with {bins} bins.")
                        st.dataframe(df.head(), use_container_width=True)

                download_link(df, "feature_engineered_dataset.csv")

            elif choice == "Data Transformation":
                st.subheader("Data Transformation")

                if st.checkbox("Log Transformation"):
                    cols_to_transform = st.multiselect(
                        "Select Columns for Log Transformation",
                        df.select_dtypes(include=[np.number]).columns.tolist(),
                    )
                    if cols_to_transform:
                        for col in cols_to_transform:
                            df[f"{col}_log"] = np.log1p(df[col])
                        st.success("Log transformation applied")
                        st.dataframe(df.head(), use_container_width=True)
                    else:
                        st.warning("Please select columns for log transformation.")

                if st.checkbox("Polynomial Features"):
                    degree = st.slider("Select Degree", 2, 5, 2)
                    numeric_features = df.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    if numeric_features:
                        poly = PolynomialFeatures(degree)
                        df_poly = pd.DataFrame(
                            poly.fit_transform(df[numeric_features]),
                            columns=poly.get_feature_names_out(numeric_features),
                        )
                        df_poly = df_poly.add_suffix(
                            "_poly"
                        )  # Avoid column name conflicts
                        df = df.join(df_poly)
                        st.success(f"Polynomial features added with degree {degree}")
                        st.dataframe(df.head(), use_container_width=True)
                    else:
                        st.warning(
                            "No numeric columns available for polynomial feature creation."
                        )

                if st.checkbox("Binning"):
                    col = st.selectbox(
                        "Select Column for Binning",
                        df.select_dtypes(include=[np.number]).columns.tolist(),
                    )
                    bins = st.slider("Number of Bins", 2, 20, 5)
                    if col:
                        df[f"{col}_binned"] = pd.cut(df[col], bins=bins)
                        st.success(f"Binning applied to {col} with {bins} bins.")
                        st.dataframe(df.head(), use_container_width=True)

                download_link(df, "transformed_dataset.csv")

            elif choice == "Model Building":
                st.subheader("Model Building")

                target = st.selectbox("Select Target Column", df.columns)
                features = df.drop(columns=[target])
                target_data = df[target]

                if st.button("Train Model"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, target_data, test_size=0.2, random_state=0
                    )
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write("Model Evaluation Report:")
                    st.text(classification_report(y_test, y_pred))

                download_link(df, "model_built_dataset.csv")

            elif choice == "Hyperparameter Tuning":
                st.subheader("Hyperparameter Tuning")

                target = st.selectbox("Select Target Column", df.columns)
                features = df.drop(columns=[target])
                target_data = df[target]

                if st.button("Hyperparameter Tuning"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, target_data, test_size=0.2, random_state=0
                    )
                    model = LogisticRegression()
                    param_grid = {"C": [0.01, 0.1, 1, 10]}
                    grid = GridSearchCV(model, param_grid, cv=3)
                    grid.fit(X_train, y_train)
                    st.write("Best Parameters:", grid.best_params_)
                    st.write("Best Score:", grid.best_score_)

                download_link(df, "tuned_model_dataset.csv")

            elif choice == "Cross-Validation":
                st.subheader("Cross-Validation")

                target = st.selectbox("Select Target Column", df.columns)
                features = df.drop(columns=[target])
                target_data = df[target]

                if st.button("K-Fold Cross-Validation"):
                    k = st.slider("Number of Folds", 2, 10, 5)
                    model = LogisticRegression()
                    kf = KFold(n_splits=k)
                    scores = []
                    for train_index, test_index in kf.split(features):
                        X_train, X_test = (
                            features.iloc[train_index],
                            features.iloc[test_index],
                        )
                        y_train, y_test = (
                            target_data.iloc[train_index],
                            target_data.iloc[test_index],
                        )
                        model.fit(X_train, y_train)
                        scores.append(model.score(X_test, y_test))
                    st.write(f"Cross-Validation Scores: {scores}")
                    st.write(f"Average Score: {np.mean(scores)}")

                download_link(df, "cross_validated_dataset.csv")

            elif choice == "Data Sampling":
                st.subheader("Data Sampling")

                if st.button("Train-Test Split"):
                    target = st.selectbox("Select Target Column", df.columns)
                    features = df.drop(columns=[target])
                    target_data = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, target_data, test_size=0.2, random_state=0
                    )
                    st.write(f"Training Data: {X_train.shape[0]} samples")
                    st.write(f"Testing Data: {X_test.shape[0]} samples")

                download_link(df, "sampled_dataset.csv")


if __name__ == "__main__":
    main()
