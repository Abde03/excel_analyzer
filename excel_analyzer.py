import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def main():
    st.set_page_config(
        page_title="Excel Data Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    with st.sidebar:
        st.markdown("### 👨‍💻 **Created By**")
        st.markdown("**Abderrahim El Azzaoui**")
        st.markdown("📧 Azzaoui03dev@gmail.com")
        st.markdown("🔗 [GitHub](https://github.com/abde03)") 
        st.markdown("💼 [LinkedIn](https://linkedin.com/in/abde1899)") 
        
        st.markdown("---")
        st.markdown("### 🛠️ **Built With**")
        st.markdown("- 🐍 Python")
        st.markdown("- 🎈 Streamlit")
        st.markdown("- 📊 Plotly")
        st.markdown("- 🐼 Pandas")
        st.markdown("- 🤖 Scikit-learn")
        
        st.markdown("---")
        st.markdown("### ⭐ **Features**")
        st.markdown("- 📁 Multi-sheet Excel support")
        st.markdown("- 📈 Interactive visualizations")
        st.markdown("- 🔬 Advanced analytics")
        st.markdown("- 🤖 Predictive modeling")
        st.markdown("- 📊 Data quality reports")
        st.markdown("- 💾 Multiple export options")
        
        st.markdown("---")
        st.markdown("### 📝 **Version**")
        st.markdown("**v2.0** - Advanced Analytics Edition")
        st.markdown("*Last updated: June 2025*")
        
        st.markdown("---")
        st.markdown("### 🆘 **Support**")
        st.markdown("Found a bug? Have suggestions?")
        st.markdown("[Report Issues](https://github.com/Abde03/excel_analyzer/issues)")
    

    st.title("📊 Excel Data Analyzer")
    st.markdown("Upload your Excel file to generate interactive charts and tables!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an Excel file", 
        type=['xlsx', 'xls'],
        help="Upload an Excel file (.xlsx or .xls format)"
    )
    
    if uploaded_file is not None:
        try:
            # Load the Excel file
            with st.spinner("Loading Excel file..."):
                # Get all sheet names
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                # Sheet selector
                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox("Select sheet:", sheet_names)
                else:
                    selected_sheet = sheet_names[0]
                
                # Load selected sheet
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            
            st.success(f"✅ Successfully loaded '{selected_sheet}' with {len(df)} rows and {len(df.columns)} columns!")
            
            # Data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
            
            # Data info
            with st.expander("ℹ️ Data Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Column Types")
                    dtype_df = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum()
                    })
                    st.dataframe(dtype_df, use_container_width=True)
                
                with col2:
                    st.subheader("Numeric Columns Summary")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    else:
                        st.info("No numeric columns found")
            
            # Generate visualizations
            st.header("📈 Data Visualizations")
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_columns) > 0:
                # Chart type selection
                chart_type = st.selectbox(
                    "Select chart type:",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"]
                )
                
                if chart_type == "Bar Chart":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis (categorical):", categorical_columns + numeric_columns)
                    with col2:
                        y_col = st.selectbox("Y-axis (numeric):", numeric_columns)
                    
                    if st.button("Generate Bar Chart"):
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Line Chart":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis:", df.columns.tolist())
                    with col2:
                        y_col = st.selectbox("Y-axis (numeric):", numeric_columns)
                    
                    if st.button("Generate Line Chart"):
                        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Scatter Plot":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_col = st.selectbox("X-axis (numeric):", numeric_columns)
                    with col2:
                        y_col = st.selectbox("Y-axis (numeric):", [col for col in numeric_columns if col != x_col])
                    with col3:
                        color_col = st.selectbox("Color by (optional):", ["None"] + categorical_columns)
                    
                    if st.button("Generate Scatter Plot"):
                        color = None if color_col == "None" else color_col
                        fig = px.scatter(df, x=x_col, y=y_col, color=color, 
                                       title=f"{y_col} vs {x_col}")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Histogram":
                    col = st.selectbox("Select column for histogram:", numeric_columns)
                    bins = st.slider("Number of bins:", 10, 100, 30)
                    
                    if st.button("Generate Histogram"):
                        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Box Plot":
                    col1, col2 = st.columns(2)
                    with col1:
                        y_col = st.selectbox("Y-axis (numeric):", numeric_columns)
                    with col2:
                        x_col = st.selectbox("Group by (optional):", ["None"] + categorical_columns)
                    
                    if st.button("Generate Box Plot"):
                        x = None if x_col == "None" else x_col
                        fig = px.box(df, x=x, y=y_col, title=f"Box Plot of {y_col}")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Correlation Heatmap":
                    if len(numeric_columns) > 1:
                        if st.button("Generate Correlation Heatmap"):
                            corr_matrix = df[numeric_columns].corr()
                            fig = px.imshow(corr_matrix, 
                                          text_auto=True, 
                                          aspect="auto",
                                          title="Correlation Heatmap")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least 2 numeric columns for correlation heatmap")
            
            else:
                st.warning("No numeric columns found for creating charts")
            
            # Data filtering and table display
            st.header("🔍 Interactive Data Table")
            
            # Filters
            with st.expander("Apply Filters"):
                filters = {}
                
                for col in categorical_columns[:5]:  # Limit to first 5 categorical columns
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) <= 20:  # Only show filter if reasonable number of options
                        selected_values = st.multiselect(
                            f"Filter by {col}:",
                            options=unique_values,
                            default=unique_values
                        )
                        filters[col] = selected_values
                
                # Apply filters
                filtered_df = df.copy()
                for col, values in filters.items():
                    if values:  # Only apply filter if values are selected
                        filtered_df = filtered_df[filtered_df[col].isin(values)]
            
            # Display filtered table
            st.subheader(f"Data Table ({len(filtered_df)} rows)")
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Interactive Dashboard
            st.header("📊 Interactive Dashboard")
            
            # Create dashboard with multiple metrics
            if len(numeric_columns) > 0:
                # Key metrics cards
                st.subheader("Key Metrics")
                
                metrics_cols = st.columns(min(4, len(numeric_columns)))
                
                for i, col in enumerate(numeric_columns[:4]):
                    with metrics_cols[i]:
                        avg_val = filtered_df[col].mean()
                        total_val = filtered_df[col].sum()
                        st.metric(
                            label=f"Avg {col}",
                            value=f"{avg_val:.2f}",
                            delta=f"Total: {total_val:.0f}"
                        )
                
                # Multi-dimensional analysis
                st.subheader("Multi-Dimensional Analysis")
                
                if len(categorical_columns) >= 2 and len(numeric_columns) >= 1:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        x_axis = st.selectbox("X-Axis:", categorical_columns, key="dash_x")
                    with col2:
                        color_by = st.selectbox("Color By:", categorical_columns, key="dash_color")
                    with col3:
                        size_by = st.selectbox("Size By:", numeric_columns, key="dash_size")
                    
                    if st.button("Generate Multi-Dimensional Chart"):
                        # Create pivot table for better visualization
                        pivot_data = filtered_df.groupby([x_axis, color_by])[size_by].sum().reset_index()
                        
                        fig = px.scatter(pivot_data, x=x_axis, y=size_by, 
                                       color=color_by, size=size_by,
                                       title=f"Multi-Dimensional Analysis: {size_by} by {x_axis} and {color_by}")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Time series analysis (if date column exists)
                date_columns = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            pd.to_datetime(df[col].head())
                            date_columns.append(col)
                        except:
                            pass
                
                if date_columns:
                    st.subheader("📅 Time Series Dashboard")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        date_col = st.selectbox("Date Column:", date_columns, key="ts_date")
                    with col2:
                        metrics = st.multiselect("Metrics to track:", numeric_columns, 
                                               default=numeric_columns[:2], key="ts_metrics")
                    
                    if metrics and st.button("Generate Time Series Dashboard"):
                        try:
                            ts_df = filtered_df.copy()
                            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
                            ts_df = ts_df.sort_values(date_col)
                            
                            # Create subplots
                            fig = make_subplots(
                                rows=len(metrics), cols=1,
                                subplot_titles=metrics,
                                shared_xaxes=True
                            )
                            
                            for i, metric in enumerate(metrics):
                                daily_data = ts_df.groupby(ts_df[date_col].dt.date)[metric].sum()
                                
                                fig.add_trace(
                                    go.Scatter(x=daily_data.index, y=daily_data.values,
                                             name=metric, mode='lines+markers'),
                                    row=i+1, col=1
                                )
                            
                            fig.update_layout(height=300*len(metrics), 
                                            title="Time Series Dashboard")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error in time series: {str(e)}")
            
            # Data Quality Report
            st.header("🔍 Data Quality Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Missing Data Analysis")
                missing_data = df.isnull().sum()
                missing_percent = (missing_data / len(df)) * 100
                
                quality_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': missing_percent.round(2)
                })
                quality_df = quality_df[quality_df['Missing Count'] > 0]
                
                if len(quality_df) > 0:
                    st.dataframe(quality_df, use_container_width=True)
                    
                    # Visualize missing data
                    fig = px.bar(quality_df, x='Column', y='Missing %',
                               title="Missing Data by Column")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing data found!")
            
            with col2:
                st.subheader("Data Types & Uniqueness")
                
                type_info = []
                for col in df.columns:
                    unique_count = df[col].nunique()
                    unique_percent = (unique_count / len(df)) * 100
                    
                    type_info.append({
                        'Column': col,
                        'Data Type': str(df[col].dtype),
                        'Unique Values': unique_count,
                        'Uniqueness %': round(unique_percent, 1)
                    })
                
                type_df = pd.DataFrame(type_info)
                st.dataframe(type_df, use_container_width=True)
            
            # Export Options
            st.header("💾 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download filtered data
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"filtered_data_{selected_sheet}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download summary report
                summary_report = f"""
DATA ANALYSIS REPORT
==================

Dataset: {selected_sheet}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

BASIC STATISTICS:
- Total Records: {len(filtered_df):,}
- Total Columns: {len(filtered_df.columns)}
- Missing Values: {filtered_df.isnull().sum().sum()}

NUMERIC COLUMNS SUMMARY:
{filtered_df[numeric_columns].describe().to_string() if numeric_columns else 'No numeric columns'}

CATEGORICAL COLUMNS:
{chr(10).join([f"- {col}: {filtered_df[col].nunique()} unique values" for col in categorical_columns]) if categorical_columns else 'No categorical columns'}
"""
                
                st.download_button(
                    label="📄 Download Report",
                    data=summary_report,
                    file_name=f"analysis_report_{selected_sheet}.txt",
                    mime="text/plain"
                )
            
            with col3:
                # Download chart data
                if len(numeric_columns) > 0:
                    chart_data = filtered_df[numeric_columns].corr()
                    csv_corr = chart_data.to_csv()
                    st.download_button(
                        label="📈 Download Correlations",
                        data=csv_corr,
                        file_name=f"correlations_{selected_sheet}.csv",
                        mime="text/csv"
                    )
            
            # Advanced Analytics
            st.header("🔬 Advanced Analytics")
            
            analytics_option = st.selectbox(
                "Choose analysis type:",
                ["Summary Statistics", "Trend Analysis", "Outlier Detection", "Group Comparison", "Predictive Insights"]
            )
            
            if analytics_option == "Summary Statistics":
                if len(numeric_columns) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Numeric Summary")
                        summary_stats = filtered_df[numeric_columns].describe()
                        st.dataframe(summary_stats, use_container_width=True)
                    
                    with col2:
                        st.subheader("Top Values")
                        for col in categorical_columns[:3]:
                            if col in filtered_df.columns:
                                st.write(f"**{col}:**")
                                top_values = filtered_df[col].value_counts().head(5)
                                st.dataframe(top_values, use_container_width=True)
            
            elif analytics_option == "Trend Analysis":
                st.subheader("📈 Trend Analysis")
                
                # Find date columns
                date_columns = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            pd.to_datetime(df[col].head())
                            date_columns.append(col)
                        except:
                            pass
                
                if date_columns and numeric_columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        date_col = st.selectbox("Date column:", date_columns)
                    with col2:
                        value_col = st.selectbox("Value column:", numeric_columns)
                    
                    if st.button("Generate Trend Analysis"):
                        try:
                            df_trend = filtered_df.copy()
                            df_trend[date_col] = pd.to_datetime(df_trend[date_col])
                            df_trend = df_trend.sort_values(date_col)
                            
                            # Daily trend
                            daily_trend = df_trend.groupby(df_trend[date_col].dt.date)[value_col].sum().reset_index()
                            fig = px.line(daily_trend, x=date_col, y=value_col, 
                                        title=f"Daily Trend: {value_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Monthly trend
                            df_trend['Month'] = df_trend[date_col].dt.to_period('M')
                            monthly_trend = df_trend.groupby('Month')[value_col].sum().reset_index()
                            monthly_trend['Month'] = monthly_trend['Month'].astype(str)
                            
                            fig2 = px.bar(monthly_trend, x='Month', y=value_col,
                                        title=f"Monthly Trend: {value_col}")
                            st.plotly_chart(fig2, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error in trend analysis: {str(e)}")
                else:
                    st.info("Need date and numeric columns for trend analysis")
            
            elif analytics_option == "Outlier Detection":
                st.subheader("🎯 Outlier Detection")
                
                if numeric_columns:
                    selected_col = st.selectbox("Select column for outlier detection:", numeric_columns)
                    
                    if st.button("Detect Outliers"):
                        # Calculate outliers using IQR method
                        Q1 = filtered_df[selected_col].quantile(0.25)
                        Q3 = filtered_df[selected_col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = filtered_df[(filtered_df[selected_col] < lower_bound) | 
                                             (filtered_df[selected_col] > upper_bound)]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Outliers", len(outliers))
                            st.metric("Outlier Percentage", f"{len(outliers)/len(filtered_df)*100:.1f}%")
                        
                        with col2:
                            st.metric("Lower Bound", f"{lower_bound:.2f}")
                            st.metric("Upper Bound", f"{upper_bound:.2f}")
                        
                        # Visualize outliers
                        fig = px.box(filtered_df, y=selected_col, title=f"Outliers in {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if len(outliers) > 0:
                            st.subheader("Outlier Records")
                            st.dataframe(outliers, use_container_width=True)
                        else:
                            st.success("No outliers detected!")
            
            elif analytics_option == "Group Comparison":
                st.subheader("👥 Group Comparison")
                
                if categorical_columns and numeric_columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        group_col = st.selectbox("Group by:", categorical_columns)
                    with col2:
                        value_col = st.selectbox("Compare values:", numeric_columns)
                    
                    if st.button("Generate Comparison"):
                        # Group statistics
                        group_stats = filtered_df.groupby(group_col)[value_col].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(2)
                        
                        st.subheader("Group Statistics")
                        st.dataframe(group_stats, use_container_width=True)
                        
                        # Visualization
                        fig = px.violin(filtered_df, x=group_col, y=value_col, 
                                      title=f"{value_col} Distribution by {group_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Top performers
                        top_groups = group_stats.sort_values('mean', ascending=False)
                        st.subheader("Top Performers (by Average)")
                        st.dataframe(top_groups.head(), use_container_width=True)
            
            elif analytics_option == "Predictive Insights":
                st.subheader("🔮 Predictive Insights")
                
                if len(numeric_columns) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("Independent variable (X):", numeric_columns)
                    with col2:
                        y_var = st.selectbox("Dependent variable (Y):", 
                                           [col for col in numeric_columns if col != x_var])
                    
                    if st.button("Generate Prediction Model"):
                        # Simple linear regression
                        from sklearn.linear_model import LinearRegression
                        from sklearn.metrics import r2_score
                        import warnings
                        warnings.filterwarnings('ignore')
                        
                        try:
                            # Prepare data
                            clean_data = filtered_df[[x_var, y_var]].dropna()
                            X = clean_data[[x_var]]
                            y = clean_data[y_var]
                            
                            # Fit model
                            model = LinearRegression()
                            model.fit(X, y)
                            
                            # Predictions
                            y_pred = model.predict(X)
                            r2 = r2_score(y, y_pred)
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R² Score", f"{r2:.3f}")
                            with col2:
                                st.metric("Slope", f"{model.coef_[0]:.3f}")
                            with col3:
                                st.metric("Intercept", f"{model.intercept_:.3f}")
                            
                            # Visualization
                            fig = px.scatter(clean_data, x=x_var, y=y_var, 
                                           title=f"Regression: {y_var} vs {x_var}")
                            
                            # Add regression line
                            x_range = np.linspace(clean_data[x_var].min(), clean_data[x_var].max(), 100)
                            y_range = model.predict(x_range.reshape(-1, 1))
                            
                            fig.add_scatter(x=x_range, y=y_range, mode='lines', 
                                          name='Regression Line', line=dict(color='red'))
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction tool
                            st.subheader("Make a Prediction")
                            input_value = st.number_input(f"Enter {x_var} value:", 
                                                        value=float(clean_data[x_var].mean()))
                            predicted_value = model.predict([[input_value]])[0]
                            st.success(f"Predicted {y_var}: {predicted_value:.2f}")
                            
                        except Exception as e:
                            st.error(f"Error in prediction model: {str(e)}")
                            st.info("Make sure you have scikit-learn installed: pip install scikit-learn")
                else:
                    st.info("Need at least 2 numeric columns for predictive analysis")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your file is a valid Excel file (.xlsx or .xls)")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an Excel file to get started!")
        
        with st.expander("ℹ️ What this app can do:"):
            st.markdown("""
            **File Processing:**
            - ✅ Read Excel files (.xlsx, .xls)
            - ✅ Handle multiple sheets
            - ✅ Display data preview and information
            
            **Visualizations:**
            - 📊 Bar charts
            - 📈 Line charts  
            - 🎯 Scatter plots
            - 📉 Histograms
            - 📦 Box plots
            - 🔥 Correlation heatmaps
            
            **Advanced Features:**
            - 🔬 Advanced Analytics (Trends, Outliers, Predictions)
            - 📊 Interactive Multi-Dimensional Dashboard
            - 📅 Time Series Analysis
            - 🔍 Data Quality Reports
            - 👥 Group Comparisons
            - 🎯 Outlier Detection
            - 🔮 Predictive Modeling (Linear Regression)
            - 💾 Multiple Export Options
            
            **Requirements:**
            ```bash
            pip install streamlit pandas plotly openpyxl xlrd scikit-learn
            ```
            """)

if __name__ == "__main__":
    main()