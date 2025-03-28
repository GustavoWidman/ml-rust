use std::path::Path;

use burn::data::dataset::Dataset;
use rand::{
    SeedableRng,
    prelude::{SliceRandom, StdRng},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

// Define the structure for an Iris data item
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IrisItem {
    #[serde(rename = "sepal length (cm)")] // Match CSV header
    pub sepal_length: f32,
    #[serde(rename = "sepal width (cm)")]
    pub sepal_width: f32,
    #[serde(rename = "petal length (cm)")]
    pub petal_length: f32,
    #[serde(rename = "petal width (cm)")]
    pub petal_width: f32,
    pub variety: String, // Target label as string initially
}

#[derive(Clone, Debug)]
pub struct IrisDataset {
    items: Vec<IrisItem>,
}

impl IrisDataset {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        Ok(Self {
            items: Self::from_csv(path, csv::ReaderBuilder::new().has_headers(true))?,
        })
    }

    pub fn random_split(mut self, ratio: f64, seed: u64) -> (IrisDataset, IrisDataset) {
        let mut rng = StdRng::seed_from_u64(seed);
        self.items.shuffle(&mut rng);
        let (train, test) = self
            .items
            .split_at((self.items.len() as f64 * ratio) as usize);

        let train = IrisDataset {
            items: train.to_vec(),
        };

        let test = IrisDataset {
            items: test.to_vec(),
        };

        (train, test)
    }

    /// Create from a csv file.
    ///
    /// The provided `csv::ReaderBuilder` can be configured to fit your csv format.
    ///
    /// The supported field types are: String, integer, float, and bool.
    ///
    /// See:
    /// - [Reading with Serde](https://docs.rs/csv/latest/csv/tutorial/index.html#reading-with-serde)
    /// - [Delimiters, quotes and variable length records](https://docs.rs/csv/latest/csv/tutorial/index.html#delimiters-quotes-and-variable-length-records)
    fn from_csv<I: Clone + DeserializeOwned, P: AsRef<Path>>(
        path: P,
        builder: &csv::ReaderBuilder,
    ) -> Result<Vec<I>, std::io::Error> {
        let mut rdr = builder.from_path(path)?;

        let mut items = Vec::new();

        for result in rdr.deserialize() {
            let item: I = result?;
            items.push(item);
        }

        Ok(items)
    }
}

// Implement the Dataset trait for Iris
impl Dataset<IrisItem> for IrisDataset {
    fn get(&self, index: usize) -> Option<IrisItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl IrisItem {
    // Helper to convert string label to integer index
    pub fn target_to_int(&self) -> i32 {
        match self.variety.as_str() {
            "Iris-setosa" => 0,
            "Iris-versicolor" => 1,
            "Iris-virginica" => 2,
            _ => panic!("Unknown Iris variety"),
        }
    }

    // Helper to get features as a fixed-size array
    pub fn features_as_array(&self) -> [f32; 4] {
        [
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]
    }
}
