// Task: create the specified collections and insert documents into them that contain all the data listed in the examples. Implement 5 queries. 
// For each collection, insert one or more documents (in addition to the examples) so that each of the 5 queries returns a non-empty result.

// To test the queries, you can use the nosql.felk.cvut.cz server.

// Domain:
// A logistics company in the Czech Republic. The system stores data about vehicles and deliveries.
// Collection vehicles:
// Each vehicle has a registration number, type (van, truck, refrigerated truck), payload capacity in kilograms, year of manufacture, and current status (active, under repair, decommissioned). 
// Assignment information is specified: depot name, city, and region. 
// The vehicle undergoes technical inspections – each inspection contains a date, mileage at the time of inspection, and a result.

// Example: vehicle with number “6BK 4470”, type “van”, payload capacity 1200 kilograms, year 2021, status “active”. Depot “Prague-Center”, city Prague, region “Hlavní město Praha”. 
// Two inspections: 5 January 2025 at mileage 34000 kilometers – passed, 5 June 2025 at mileage 52000 kilometers – passed.

// Collection deliveries:
// Each delivery contains the departure date, arrival date, distance in kilometers, and cost in korunas. 
// Route information is specified: departure city, destination city, and cargo type (standard, fragile, perishable). 
// The vehicle registration number, driver name, and a list of cargo items with item name, weight, and number of packages are also specified.

// Example: delivery departed on 28 June 2025, arrived on 28 June 2025, distance 45 kilometers, cost 1800 korunas. 
// Route: from Prague to Kladno, cargo type “perishable”. 
// Vehicle “6BK 4470”, driver Tomáš Král. 
// Cargo: 12 boxes of dairy products at 8 kilograms each, 4 crates of fruit at 18 kilograms each.

// Queries:
// Query 1:
// Due to increased fuel costs, adjust pricing for long-distance routes. Increase the cost by 7 percent for all deliveries with a distance greater than 300 kilometers.
// Query 2:
// Find all deliveries with cargo type “perishable” with an arrival date in July 2025. Sort by cost in descending order. Output the arrival date, departure city, and cost.
// Query 3:
// Find all vehicles that have at least one inspection record with result 'passed' and mileage greater than or equal to 100,000 km. 
// Sort the vehicles by year of manufacture in ascending order. Output the registration number, vehicle type, and year of manufacture.
// Query 4:
// For each driver, calculate the number of deliveries with a departure date in 2025 and the average delivery cost. 
// Exclude drivers with fewer than 2 such deliveries. 
// Sort by the number of deliveries in descending order. Output the driver name, number of deliveries, and average cost rounded to a whole number.
// Query 5:
// Find the top five cargo item names by total shipped weight for deliveries with a departure date in 2025 (sum of item weight multiplied by number of packages). 
// Output the cargo item name, total shipped weight, and the number of deliveries in which it appears. Sort by total shipped weight in descending order.

// Date/Time storage requirement: All date and date-time attributes (e.g., inspection dates, order timestamps, rental start/end) must be stored as MongoDB BSON Date values (ISODate), not as strings. All timestamps must use a consistent timezone (recommended: UTC). If a task requires time arithmetic (e.g., duration in minutes), start and end must be stored as Date values so that the difference can be computed correctly.

// RIEŠENIE
// db.vehicles
// - registration number (string),
// - type (van, truck, refrigerated truck) (string),
// - payload_kg (number),
// - year of manufacture (number),
// - current status (active, under repair, decommissioned) (string)
// - depot (object)
// --- depot name (string)
// --- city (string)
// --- region (string)
// - technical inspections (array of objects) 
// --- inspection (object)
// ------ date (data)
// ------ mileage at the time of inspection (number)
// ------ result (string)

// db.deliveries
// - the departure date (data) -- 28 June 2025,
// - arrival date (date) -- 28 June 2025,
// - distance in kilometers (number) -- 45 kilometers,
// - cost in korunas (number) -- cost 1800 korunas,
// - Route (object)
// --- departure city (string) -- Prague,
// --- destination city (string) -- Kladno,
// --- cargo type (standard, fragile, perishable) (string) -- “perishable”, 
// - The vehicle registration number (string) -- “6BK 4470”,
// - driver name (string) -- Tomáš Král,
// - items (array of objects),
// --- item (object)
// ----- item name (string) -- box of dairy products, 
// ----- weight (number) -- 8 kilograms
// ----- number of packages are also specified (number) -- 12,

// Query 1
db.deliveries.updateMany(
    {"distance": {
        $gt: 300
    }},
    {$set: {
        "cost": {
            $multiply: ["$cost", 1.07]
        }
    }}
)

// Query 2:
// Find all deliveries with cargo type “perishable” with an arrival date in July 2025. Sort by cost in descending order. Output the arrival date, departure city, and cost.
db.deliveries.find({
    $and: [
        {"route.type": {
            $eq: "perishable"
        }},
        {"arrival_date": {
            $and: [
                {$gte: ISODate("2025-07-01T00:00:00Z")},
                {$lt: ISODate("2025-08-01T00:00:00Z")}
            ]
        }}
    ]
}, {
    "arrival_date": 1,
    "route.departure_city": 1,
    "cost": 1,
    "_id": 0
}).sort({"cost": -1});

// Alebo
db.deliveries.find({
    "route.type": "perishable",
    "arrival_date": {
        $gte: ISODate("2025-07-01T00:00:00Z"),
        $lt: ISODate("2025-08-01T00:00:00Z")
    }
}, {
    "arrival_date": 1,
    "route.departure_city": 1,
    "cost": 1,
    "_id": 0
}).sort({"cost": -1}); 

// Query 3:
// Find all vehicles that have at least one inspection record with result 'passed' and mileage greater than or equal to 100,000 km. 
// Sort the vehicles by year of manufacture in ascending order. Output the registration number, vehicle type, and year of manufacture.
db.vehicles.find({
    "inspections": {
        $elemMatch: {
            "result": "passed",
            "milage": { $gte: 100000 }
        }
    }
}, {
    "registration number": 1, 
    "vehicle type": 1,
    "year of manufacture": 1,
    "_id": 0
}).sort({"year": 1});

// Query 4:
// For each driver, calculate the number of deliveries with a departure date in 2025 and the average delivery cost. 
// Exclude drivers with fewer than 2 such deliveries.
db.deliveries.aggregate([
    {
        $match: {
            "departure_date": {
                $gte: ISODate("2025-01-01T00:00:00Z"),
                $lt: ISODate("2026-01-01T00:00:00Z")
            }
        }
    },
    {
        $group: {
            "_id": "$driver_name",
            "avg": { $avg: "$price" },
            "count": { $sum: 1 }
        }
    },
    {
        $match: {
            "count": { $gte: 2 }
        }
    }
]);


// db.deliveries
// - the departure date (data) -- 28 June 2025,
// - arrival date (date) -- 28 June 2025,
// - distance in kilometers (number) -- 45 kilometers,
// - cost in korunas (number) -- cost 1800 korunas,
// - Route (object)
// --- departure city (string) -- Prague,
// --- destination city (string) -- Kladno,
// --- cargo type (standard, fragile, perishable) (string) -- “perishable”, 
// - The vehicle registration number (string) -- “6BK 4470”,
// - driver name (string) -- Tomáš Král,
// - items (array of objects),
// --- item (object)
// ----- item name (string) -- box of dairy products, 
// ----- weight (number) -- 8 kilograms
// ----- number of packages are also specified (number) -- 12,

// 12 boxes of dairy products at 8 kilograms each, 4 crates of fruit at 18 kilograms each.

// {
//     delivery1
//     [
//         {"name1", 8, 12},
//         {"name2", 4, 18},
        
//     ]
// }

// {
//     delivery2
//     [
//         {"name1", 8, 12},
//         {"name2", 4, 18},
        
//     ]
// }

// {
//     delivery1
//     {"name1", 8, 12},
//     {"name2", 4, 18},
// }

// {
//     delivery2
//     {"name1", 8, 12},
//     {"name2", 4, 18},
        
// }



// Query 5:
// Find the top five cargo item names by total shipped weight for deliveries with a departure date in 2025 (sum of item weight multiplied by number of packages). 
// Output the cargo item name, total shipped weight, and the number of deliveries in which it appears. Sort by total shipped weight in descending order.
db.deliveries.aggregate([
    {
        $match: {
            "departure_date": {
                $gte: ISODate("2025-01-01T00:00:00Z"),
                $lt: ISODate("2026-01-01T00:00:00Z")
            }
        }
    },
    { $unwind: "$cargo_items" }, 
    {
        $addFields: {
            "item_total_weight": { $multiply: ["$cargo_items.weight_kg", "$cargo_items.packages_count"] }
        }
    },
    {
        $group: {
            "_id": "$cargo_items.item_name",
            "total_weight_final": { $sum: "$item_total_weight" }, // Added comma here
            "delivery_appearance_count": { $sum: 1 }
        } 
    },
    { $sort: { "total_weight_final": -1 } },
    { $limit: 5 },
    {
        $project: {
            "_id": 0,
            "item_name": "$_id",
            "total_weight_final": 1,
            "delivery_appearance_count": 1
        }
    }
]);

db.deliveries.aggregate([
  // 1. Filter: Deliveries in the year 2025
  {
    $match: {
      "departure_date": {
        $gte: ISODate("2025-01-01T00:00:00Z"),
        $lt: ISODate("2026-01-01T00:00:00Z")
      }
    }
  },

  // 2. Deconstruct: Turn each array element into its own document
  { $unwind: "$cargo_items" },

  // 3. Group: Calculate weight and count occurrences
  {
    $group: {
      "_id": "$cargo_items.item_name",
      "total_shipped_weight": { 
        $sum: { $multiply: ["$cargo_items.weight_kg", "$cargo_items.packages_count"] } 
      },
      "delivery_appearance_count": { $sum: 1 }
    }
  },

  // 4. Sort: Highest weight first
  { $sort: { "total_shipped_weight": -1 } },

  // 5. Limit: Only the top five
  { $limit: 5 },

  // 6. Project: Clean up the field names for the output
  {
    $project: {
      "_id": 0,
      "item_name": "$_id",
      "total_shipped_weight": 1,
      "delivery_appearance_count": 1
    }
  }
]);
