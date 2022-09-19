from dataclasses import dataclass


@dataclass
class COLS:
    DROP_COL = []
    ID = "Id"
    TARGET = "SalePrice"

    # breakdown of features
    NUMERIC_COLS = ["LotFrontage",
                    "LotArea",
                    "MasVnrArea",
                    "BsmtFinSF1",
                    "BsmtFinSF2",
                    "BsmtUnfSF",
                    "TotalBsmtSF",
                    "1stFlrSF",
                    "2ndFlrSF",
                    "LowQualFinSF",
                    "GrLivArea",
                    "GarageArea",
                    "GarageCars",
                    "WoodDeckSF",
                    "OpenPorchSF",
                    "EnclosedPorch",
                    "3SsnPorch",
                    "ScreenPorch",
                    "PoolArea",
                    "MiscVal",
                    "MoSold",
                    "BedroomAbvGr",
                    "KitchenAbvGr",
                    "FullBath",
                    "BsmtFullBath",
                    "Fireplaces",
                    "BsmtHalfBath",
                    "HalfBath",
                    "TotRmsAbvGrd"
                    ]

    CATEGORICAL_COLS = ["MSSubClass",
                        "MSZoning",
                        "Street",
                        "Alley",
                        "LotShape",
                        "LandContour",
                        "Utilities",
                        "LotConfig",
                        "LandSlope",
                        "Neighborhood",
                        "Condition1",
                        "Condition2",
                        "BldgType",
                        "HouseStyle",
                        "OverallQual",
                        "OverallCond",
                        "YearBuilt",
                        "YearRemodAdd",
                        "RoofStyle",
                        "RoofMatl",
                        "Exterior1st",
                        "Exterior2nd",
                        "MasVnrType",
                        "ExterQual",
                        "ExterCond",
                        "Foundation",
                        "BsmtQual",
                        "BsmtCond",
                        "BsmtExposure",
                        "BsmtFinType1",
                        "BsmtFinType2",
                        "Heating",
                        "HeatingQC",
                        "CentralAir",
                        "Electrical",
                        "KitchenQual",
                        "Functional",
                        "FireplaceQu",
                        "GarageType",
                        "GarageYrBlt",
                        "GarageFinish",
                        "GarageQual",
                        "GarageCond",
                        "PavedDrive",
                        "PoolQC",
                        "Fence",
                        "MiscFeature",
                        "SaleType",
                        "SaleCondition",
                        "YrSold",
                        ]

    # breakdown of categorical columns
    BASE_YEAR = "YrSold"

    DATE_COLS = ["YearBuilt",
                 "YearRemodAdd",
                 "GarageYrBlt"
                 ]

    NOMINAL_COLS = [
        "MSSubClass",
        "MSZoning",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "Foundation",
        "Heating",
        "CentralAir",
        "Electrical",
        "GarageType",
        "PavedDrive",
        "MiscFeature",
        "SaleType",
        "SaleCondition",
    ]

    ORDINAL_COLS = [
        "OverallQual",
        "OverallCond",
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "HeatingQC",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence"
    ]

    NEW_FEATURES = ['totalSqFeet',
                    'TotalBsmtSF'
                    ]
