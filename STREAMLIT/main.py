import streamlit as st
import folium
import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

def main():
    # Load the Excel files into pandas dataframes
    manufacturers_df = pd.read_excel("ManufacturersLocation.xlsx")
    plants_df = pd.read_excel("PlantsLocation.xlsx")
    aluminum_df = pd.read_csv('aluminum.csv',sep=";")
    titanium_df = pd.read_csv('titanium.csv', sep=";")
    steel_df = pd.read_csv('steel.csv', sep=";")
    special_alloys_df = pd.read_csv('special_alloys.csv', sep=";")
    composites_df = pd.read_csv('composites.csv', sep=";")
    aluminum_stocks = pd.read_csv('AluminumStocks.csv',sep=",")
    titanium_stocks = pd.read_csv('TitaniumStocks.csv', sep=",")
    steel_stocks = pd.read_csv('SteelStocks.csv', sep=",")
    special_alloys_stocks = pd.read_csv('CopperStocks.csv', sep=",")
    composites_stocks = pd.read_csv('CompositesStocks.csv', sep=",")

    # Create tabs in the sidebar
    tabs = st.sidebar.radio("Menu", ("Failure Predictor", "Products", "Plants", "Manufacturers", "Prices"))

    # Display content based on selected tab
    if tabs == "Failure Predictor":
        st.header("Predicting line production performance")
        st.subheader("Please, enter the inputs needed")
        # Make the model
        df = pd.read_csv('predictive_maintenance.csv', sep=',')
        df = df.drop(["UDI", "Product ID", "Type"], axis=1, errors="ignore")
        df_binary = df.drop("Failure Type", axis=1, errors="ignore")

        X_binary = df_binary.drop("Target", axis=1)
        y_binary = df_binary["Target"]
        X_binary_train, X_binary_test, y_binary_train, y_binary_test = train_test_split(X_binary, y_binary,
                                                                                        test_size=0.2, random_state=42)

        to_scale = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
                    "Tool wear [min]"]
        sc = StandardScaler()
        X_binary_train[to_scale] = sc.fit_transform(X_binary_train[to_scale])
        X_binary_test[to_scale] = sc.transform(X_binary_test[to_scale])

        model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42).fit(X_binary_train,
                                                                                              y_binary_train)

        # Input boxes for data
        air_temperature = st.number_input("Air Temperature [K]", value=300.0, step=1.0)
        process_temperature = st.number_input("Process Temperature [K]", value=300.0, step=1.0)
        rotational_speed = st.number_input("Rotational Speed [rpm]", value=1000, step=10)
        torque = st.number_input("Torque [Nm]", value=50.0, step=1.0)
        tool_wear = st.number_input("Tool Wear [min]", value=0, step=1)

        # Display the input data
        data = {
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [process_temperature],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear]
        }
        df_user_input = pd.DataFrame(data)
        st.subheader("User Input:")
        st.dataframe(df_user_input)

        # Make prediction
        user_input_scaled = sc.transform(df_user_input[to_scale])
        user_prediction = model.predict(user_input_scaled)[0]
        prediction_score = model.predict_proba(user_input_scaled)[0][user_prediction]

        # Multiclass model
        df_multi = df.drop("Target", axis=1, errors="ignore")
        X_multi = df_multi.drop("Failure Type", axis=1)
        y_multi = df_multi["Failure Type"]

        X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi, y_multi, test_size=0.2,random_state=42)
        ohe = OneHotEncoder()
        y_multi_train = ohe.fit_transform(y_multi_train.values.reshape(-1, 1))
        y_multi_test = ohe.transform(y_multi_test.values.reshape(-1, 1))
        y_multi_train = pd.DataFrame(y_multi_train.toarray(), columns=ohe.categories_)
        y_multi_test = pd.DataFrame(y_multi_test.toarray(), columns=ohe.categories_)
        to_scale = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]","Tool wear [min]"]
        sc = StandardScaler()
        X_multi_train[to_scale] = sc.fit_transform(X_multi_train[to_scale])
        X_multi_test[to_scale] = sc.transform(X_multi_test[to_scale])
        model2 = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42).fit(X_multi_train,y_multi_train)
        # Make prediction with multiclass model
        user_input_scaled_multi = sc.transform(df_user_input[to_scale])
        user_prediction2 = model2.predict(user_input_scaled_multi)
        user_prediction2_index = user_prediction2.argmax()  # Get the index of the highest probability (predicted class)
        user_prediction2_label = ohe.categories_[0][user_prediction2_index]  # Get the corresponding category label

        # Display the prediction results
        st.subheader("Prediction:")
        if user_prediction == 0:
            st.warning("There should be no failure.")
        elif user_prediction == 1:
            # Display the cause of the failure
            st.error(f"There is a {user_prediction2_label}")
            # Display the corresponding image for the failure
            failure_image_path = "warning.png"  # Replace with the correct image path
            image = Image.open(failure_image_path)
            st.image(image, caption=f"Failure Type: {user_prediction2_label}", use_column_width=False)




    elif tabs == "Products":
        st.header("Products Info and Criticality")
        # Convert the 'Date' column to a pandas datetime object
        # Add button in the sidebar
        if st.sidebar.button("Input purchase order"):
            # Show a pop-up with input fields
            user_input = st.sidebar.text_input("Enter ordered quantity")
            submit_button = st.sidebar.button("Submit")

            if submit_button:
                # Process the input data
                st.sidebar.success(f"Hello, {user_input}!")

        if st.sidebar.button("Input Consumption"):
            # Show a pop-up with input fields
            user_input = st.sidebar.text_input("Enter consumed quantity")
            submit_button = st.sidebar.button("Submit")

            if submit_button:
                # Process the input data
                st.sidebar.success(f"Hello, {user_input}!")
        # Create a slider widget to select the material to visualize
        material_options = ["Aluminum", "Steel", "Special Alloys", "Titanium", "Composites"]
        selected_material = st.sidebar.selectbox("Select Material", material_options)

        # Select the appropriate DataFrame based on the selected material
        if selected_material == "Aluminum":
            selected_data = aluminum_df
        elif selected_material == "Steel":
            selected_data = steel_df
        elif selected_material == "Special Alloys":
            selected_data = special_alloys_df
        elif selected_material == "Titanium":
            selected_data = titanium_df
        elif selected_material == "Composites":
            selected_data = composites_df
        else:
            selected_data = pd.DataFrame()

        # Convert the 'Date' column to a pandas datetime object and set it as the index
        selected_data['Date'] = pd.to_datetime(selected_data['Date'])
        selected_data.set_index('Date', inplace=True)

        # Perform SARIMA forecasting for the next 6 months of consumptions
        consumptions_series = selected_data['Consumptions']
        order = (1, 1, 1)  # Order (p, d, q)
        seasonal_order = (1, 1, 1, 12)  # Seasonal order (P, D, Q, S)
        sarima_model = SARIMAX(consumptions_series, order=order, seasonal_order=seasonal_order)
        sarima_fit = sarima_model.fit()

        # Forecast the next 6 months of consumptions
        forecasted_consumptions = sarima_fit.forecast(steps=25)

        # Create a DataFrame with the forecasted consumptions and corresponding dates
        forecast_dates = pd.date_range(selected_data.index[-1], periods=26, freq='W')[1:]  # Start from the next week
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Consumptions': forecasted_consumptions})

        # Calculate the forecasted stock by subtracting the forecasted consumptions from the historical stock
        forecast_df['Stock'] = selected_data['Stock'].iloc[-1] - forecast_df['Forecasted_Consumptions'].cumsum()

        # Add a new column to indicate whether it's historical or forecasted data
        selected_data['Type'] = 'Historical'
        forecast_df['Type'] = 'Forecasted'

        # Combine historical and forecasted stock data
        combined_data = pd.concat([selected_data[['Stock', 'Type']], forecast_df[['Stock', 'Type']]], axis=0)

        # Convert the 'Date' column to a pandas datetime object in combined_data
        combined_data['Date'] = pd.to_datetime(combined_data.index)

        # Display the current stock
        st.subheader(f'Current {selected_material} Stock')
        st.write(selected_data['Stock'][-1:])

        # Function to check stock forecast and display appropriate message
        def check_stock_forecast(stock_forecast):
            if stock_forecast.min() >= 0:
                st.success("Stock is sufficient for the next 6 months.")
            elif stock_forecast.head(13).min() >= 0:
                st.warning("Order is needed within the next 6 months. Material will run out.")
            else:
                st.error("Order is needed. You will run out of material in less than 3 months.")

        # Check if the forecasted stock needs an order
        check_stock_forecast(forecast_df['Stock'])

        # Create a line chart using Altair for the combined stock data
        chart = alt.Chart(combined_data.reset_index()).mark_line().encode(
            x='Date:T',
            y='Stock',
            tooltip=['Date:T', 'Stock'],
            color=alt.Color('Type:N', legend=alt.Legend(title=None))  # Use Type column for color encoding
        ).properties(
            width=1000,
            height=400,
            title=f'{selected_material} Historical and Forecasted Stock for the Next 6 Months'
        )

        # Display the chart in Streamlit app
        st.title(f'{selected_material} Material Forecast')
        st.altair_chart(chart, use_container_width=False)

        # Display the forecasted stock in a table
        st.subheader('Forecasted Stock for the Next 6 Months')
        st.write(forecast_df[['Date', 'Stock']].set_index('Date'))


    elif tabs == "Plants":
        st.header("Plants Overview")
        st.subheader("Please, find all the additional information in the table below" )
        # Add content specific to Tab 3

        # Add content specific to Plants
        plants_df["Lat"] = pd.to_numeric(plants_df["Lat"], errors="coerce")
        plants_df["Long"] = pd.to_numeric(plants_df["Long"], errors="coerce")

        # Remove rows with missing or invalid latitude and longitude values
        plants_df = plants_df.dropna(subset=["Lat", "Long"])

        # Create a folium map centered on a specific location
        plant_map = folium.Map(
            location=[plants_df["Lat"].mean(), plants_df["Long"].mean()], zoom_start=2)

        # Add markers for manufacturers
        for index, row in plants_df.iterrows():
            folium.Marker(location=[row["Lat"], row["Long"]],
                          popup=row["Location"],
                          icon=folium.Icon(color="darkblue", icon="home")).add_to(plant_map)

        # Display the map
        st.write(plant_map._repr_html_(), unsafe_allow_html=True)

        plants_info = plants_df.drop(['Lat', 'Long'], axis=1)
        # Display the DataFrame in Streamlit app
        st.subheader("Plants Information")
        st.dataframe(plants_info)

    elif tabs == "Manufacturers":
        st.header("Manufacturers Overview")
        st.subheader("In case an order is needed, please check the table below")
        # Add content specific to Manufacturers

        manufacturers_df["Lat"] = pd.to_numeric(manufacturers_df["Lat"], errors="coerce")
        manufacturers_df["Long"] = pd.to_numeric(manufacturers_df["Long"], errors="coerce")

        # Remove rows with missing or invalid latitude and longitude values
        manufacturers_df = manufacturers_df.dropna(subset=["Lat", "Long"])

        # Create a folium map centered on a specific location
        manufacturer_map = folium.Map(
            location=[manufacturers_df["Lat"].mean(), manufacturers_df["Long"].mean()], zoom_start=2)

        # Add markers for manufacturers
        for index, row in manufacturers_df.iterrows():
            folium.Marker(location=[row["Lat"], row["Long"]],
                          popup=row["Location"],
                          icon=folium.Icon(color="green", icon="shopping-cart")).add_to(manufacturer_map)

        # Display the map
        st.write(manufacturer_map._repr_html_(), unsafe_allow_html=True)

        manufacturers_info = manufacturers_df.drop(['Lat', 'Long'], axis=1)
        # Display the DataFrame in Streamlit app
        st.subheader("Manufacturers Information")
        st.dataframe(manufacturers_info)

    elif tabs == "Prices":
        st.header("Price Index")
        # Add content specific to Tab 5
        # Create a slider widget to select the material to visualize
        material_options = ["Aluminum", "Steel", "Special Alloys", "Titanium", "Composites"]
        selected_material = st.sidebar.selectbox("Select Material", material_options)

        # Select the appropriate DataFrame based on the selected material
        if selected_material == "Aluminum":
            selected_data = aluminum_stocks
        elif selected_material == "Steel":
            selected_data = steel_stocks
        elif selected_material == "Special Alloys":
            selected_data = special_alloys_stocks
        elif selected_material == "Titanium":
            selected_data = titanium_stocks
        elif selected_material == "Composites":
            selected_data = composites_stocks
        else:
            selected_data = pd.DataFrame()

        # Convert the 'Date' column to datetime type for proper plotting
        selected_data['DATE'] = pd.to_datetime(selected_data['DATE'])

        # Create a line chart using Altair for the combined stock data
        chart = alt.Chart(selected_data.reset_index()).mark_line().encode(
            x='DATE',
            y='INDEX_PRICE',
            tooltip=['DATE', 'INDEX_PRICE'],
        ).properties(
            width=800,
            height=400,
            title=f'{selected_material} Index Price over Time'
        )

        # Display the chart in Streamlit app
        st.title(f'{selected_material} Raw Materials')
        st.altair_chart(chart, use_container_width=False)


if __name__ == "__main__":
    main()







