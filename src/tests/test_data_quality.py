import great_expectations as ge
from kedro.io import DataCatalog
import yaml

def test_data_quality():
    # Load the configuration from the YAML file
    with open('conf/base/catalog.yml', 'r') as file:
        catalog_config = yaml.safe_load(file)

    # Create the DataCatalog object
    catalog = DataCatalog.from_config(catalog_config)

    # Load the dataset
    data = catalog.load("raw_data")

    # Convert to a Great Expectations dataset
    ge_data = ge.from_pandas(data)

    # Define the expectations
    ge_data.expect_column_to_exist("col_name")

    # Run the tests
    results = ge_data.validate()

    # Print results
    print(results)

    # Run the test function


if __name__ == "__main__":
    test_data_quality()
