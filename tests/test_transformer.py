import pandas as pd
from pier_ds_utils import transformer


def test_custom_discrete_categorizer():
    categorizer = transformer.CustomDiscreteCategorizer(
        column='gender',
        categories=[
            ['M', 'm', 'Masculino', 'masculino'],
            ['F', 'f', 'Feminino', 'feminino'],
        ],
        labels=['M', 'F'],
        default_value='M',
    )

    X = pd.DataFrame(
        {
            'gender': [
                'M', 
                'm',
                'Masculino',
                'masculino',
                'F',
                'f', 
                'Feminino',
                'feminino',
                '',
                'non-sense',
                None,
                42,
                42.42,
            ],
        }
    )

    X_transformed = categorizer.fit_transform(X)

    assert X_transformed['gender'].tolist() == [
        'M',
        'M',
        'M',
        'M',
        'F',
        'F',
        'F',
        'F',
        'M',
        'M',
        'M',
        'M',
        'M'
    ]


def test_custom_interval_categorizer():
    categorizer = transformer.CustomIntervalCategorizer(
        column='price',
        intervals=[
            (498, 2700),
            (2700, 3447.6),
            (3447.6, 5592),
            (5592, 13950),
        ],
        labels=['fx1_apple', 'fx2_apple', 'fx3_apple', 'fx4_apple'],
        default_value='fx_outras_marcas',
        output_column='price_fx',
    )

    X = pd.DataFrame(
        {
            'price': [
                498,
                2699,
                2700,
                3447.5,
                3447.6,
                5591,
                5592,
                13949,
                200,
                15999,
            ],
        }
    )

    X = categorizer.fit_transform(X)

    assert X['price_fx'].tolist() == [
        'fx1_apple',
        'fx1_apple',
        'fx2_apple',
        'fx2_apple',
        'fx3_apple',
        'fx3_apple',
        'fx4_apple',
        'fx4_apple',
        'fx_outras_marcas',
        'fx_outras_marcas',
    ]

def test_custom_interval_categorizer_by_category():
    categorizer = transformer.CustomIntervalCategorizerByCategory(
        category_column='brand',
        interval_categorizers={
            'apple': transformer.CustomIntervalCategorizer(
                column='price',
                intervals=[
                    (498, 2700),
                    (2700, 3447.6),
                    (3447.6, 5592),
                    (5592, 13950),
                ],
                labels=['fx1_apple', 'fx2_apple', 'fx3_apple', 'fx4_apple'],
            ),
            'samsung': transformer.CustomIntervalCategorizer(
                column='price',
                intervals=[
                    (189, 1500),
                    (1500, 11340),
                ],
                labels=['fx1_samsung', 'fx2_samsung'],
            )
        },
        default_categorizer=transformer.CustomIntervalCategorizer(
                column='price',
                intervals=[(240, 5260)],
                labels=['fx_outras_marcas'],
        ),
        output_column='price_fx',
    )

    X = pd.DataFrame(
        {
            'brand': [
                'apple',
                'apple',
                'apple',
                'apple',
                'apple',
                'apple',
                'apple',
                'apple',
                'samsung',
                'samsung',
                'samsung',
                'samsung',
                'outras_marcas',
                'outras_marcas',
            ],
            'price': [
                498,
                2699,
                2700,
                3447.5,
                3447.6,
                5591,
                5592,
                13949,
                189,
                1499,
                1500,
                11339,
                240,
                5259,
            ],
        }
    )

    X = categorizer.fit_transform(X)

    assert X['price_fx'].tolist() == [
        'fx1_apple',
        'fx1_apple',
        'fx2_apple',
        'fx2_apple',
        'fx3_apple',
        'fx3_apple',
        'fx4_apple',
        'fx4_apple',
        'fx1_samsung',
        'fx1_samsung',
        'fx2_samsung',
        'fx2_samsung',
        'fx_outras_marcas',
        'fx_outras_marcas',
    ]
