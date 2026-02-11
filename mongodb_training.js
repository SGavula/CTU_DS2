db.restaurants.insertMany([
  {"address": {"building": "1007", "coord": [-73.856077, 40.848447], "street": "Morris Park Ave", "zipcode": "10462"}, "borough": "Bronx", "cuisine": "Bakery", "grades": [{"date": {"$date": 1393804800000}, "grade": "A", "score": 2}, {"date": {"$date": 1378857600000}, "grade": "A", "score": 6}, {"date": {"$date": 1358985600000}, "grade": "A", "score": 10}, {"date": {"$date": 1322006400000}, "grade": "A", "score": 9}, {"date": {"$date": 1299715200000}, "grade": "B", "score": 14}], "name": "Morris Park Bake Shop", "restaurant_id": "30075445"},
  {"address": {"building": "469", "coord": [-73.961704, 40.662942], "street": "Flatbush Avenue", "zipcode": "11225"}, "borough": "Brooklyn", "cuisine": "Hamburgers", "grades": [{"date": {"$date": 1419897600000}, "grade": "A", "score": 8}, {"date": {"$date": 1404172800000}, "grade": "B", "score": 23}, {"date": {"$date": 1367280000000}, "grade": "A", "score": 12}, {"date": {"$date": 1336435200000}, "grade": "A", "score": 12}], "name": "Wendy'S", "restaurant_id": "30112340"},
  {"address": {"building": "351", "coord": [-73.98513559999999, 40.7676919], "street": "West   57 Street", "zipcode": "10019"}, "borough": "Manhattan", "cuisine": "Irish", "grades": [{"date": {"$date": 1409961600000}, "grade": "A", "score": 2}, {"date": {"$date": 1374451200000}, "grade": "A", "score": 11}, {"date": {"$date": 1343692800000}, "grade": "A", "score": 12}, {"date": {"$date": 1325116800000}, "grade": "A", "score": 12}], "name": "Dj Reynolds Pub And Restaurant", "restaurant_id": "30191841"},
  {"address": {"building": "2780", "coord": [-73.98241999999999, 40.579505], "street": "Stillwell Avenue", "zipcode": "11224"}, "borough": "Brooklyn", "cuisine": "American ", "grades": [{"date": {"$date": 1402358400000}, "grade": "A", "score": 5}, {"date": {"$date": 1370390400000}, "grade": "A", "score": 7}, {"date": {"$date": 1334275200000}, "grade": "A", "score": 12}, {"date": {"$date": 1318377600000}, "grade": "A", "score": 12}], "name": "Riviera Caterer", "restaurant_id": "40356018"},
  {"address": {"building": "97-22", "coord": [-73.8601152, 40.7311739], "street": "63 Road", "zipcode": "11374"}, "borough": "Queens", "cuisine": "Jewish/Kosher", "grades": [{"date": {"$date": 1416787200000}, "grade": "Z", "score": 20}, {"date": {"$date": 1358380800000}, "grade": "A", "score": 13}, {"date": {"$date": 1343865600000}, "grade": "A", "score": 13}, {"date": {"$date": 1323907200000}, "grade": "B", "score": 25}], "name": "Tov Kosher Kitchen", "restaurant_id": "40356068"},
  {"address": {"building": "8825", "coord": [-73.8803827, 40.7643124], "street": "Astoria Boulevard", "zipcode": "11369"}, "borough": "Queens", "cuisine": "American ", "grades": [{"date": {"$date": 1416009600000}, "grade": "Z", "score": 38}, {"date": {"$date": 1398988800000}, "grade": "A", "score": 10}, {"date": {"$date": 1362182400000}, "grade": "A", "score": 7}, {"date": {"$date": 1328832000000}, "grade": "A", "score": 13}], "name": "Brunos On The Boulevard", "restaurant_id": "40356151"},
  {"address": {"building": "2206", "coord": [-74.1377286, 40.6119572], "street": "Victory Boulevard", "zipcode": "10314"}, "borough": "Staten Island", "cuisine": "Jewish/Kosher", "grades": [{"date": {"$date": 1412553600000}, "grade": "A", "score": 9}, {"date": {"$date": 1400544000000}, "grade": "A", "score": 12}, {"date": {"$date": 1365033600000}, "grade": "A", "score": 12}, {"date": {"$date": 1327363200000}, "grade": "A", "score": 9}], "name": "Kosher Island", "restaurant_id": "40356442"},
  {"address": {"building": "7114", "coord": [-73.9068506, 40.6199034], "street": "Avenue U", "zipcode": "11234"}, "borough": "Brooklyn", "cuisine": "Delicatessen", "grades": [{"date": {"$date": 1401321600000}, "grade": "A", "score": 10}, {"date": {"$date": 1389657600000}, "grade": "A", "score": 10}, {"date": {"$date": 1375488000000}, "grade": "A", "score": 8}, {"date": {"$date": 1342569600000}, "grade": "A", "score": 10}, {"date": {"$date": 1331251200000}, "grade": "A", "score": 13}, {"date": {"$date": 1318550400000}, "grade": "A", "score": 9}], "name": "Wilken'S Fine Food", "restaurant_id": "40356483"},
  {"address": {"building": "6409", "coord": [-74.00528899999999, 40.628886], "street": "11 Avenue", "zipcode": "11219"}, "borough": "Brooklyn", "cuisine": "American ", "grades": [{"date": {"$date": 1405641600000}, "grade": "A", "score": 12}, {"date": {"$date": 1375142400000}, "grade": "A", "score": 12}, {"date": {"$date": 1360713600000}, "grade": "A", "score": 11}, {"date": {"$date": 1345075200000}, "grade": "A", "score": 2}, {"date": {"$date": 1313539200000}, "grade": "A", "score": 11}], "name": "Regina Caterers", "restaurant_id": "40356649"},
  {"address": {"building": "1839", "coord": [-100.9482609, 40.6408271], "street": "Nostrand Avenue", "zipcode": "11226"}, "borough": "Brooklyn", "cuisine": "Ice Cream, Gelato, Yogurt, Ices", "grades": [{"date": {"$date": 1405296000000}, "grade": "A", "score": 12}, {"date": {"$date": 1373414400000}, "grade": "A", "score": 8}, {"date": {"$date": 1341964800000}, "grade": "A", "score": 5}, {"date": {"$date": 1329955200000}, "grade": "A", "score": 8}], "name": "Taste The Tropics Ice Cream", "restaurant_id": "40356731"},
])


// db.restaurants.find()
// db.restaurants.find({}, {
//  restaurant_id: 1, name: 1, borough: 1, cuisine: 1
// })

// db.restaurants.find({}, {
//   _id: 0, restaurant_id: 1, name: 1, borough: 1, "address.zipcode": 1
// })

// db.restaurants.find({
//   "borough": "Bronx"
// })

// db.restaurants.find({
//   "borough": "Bronx"
// }).skip(5).limit(5)

// db.restaurants.find({
//   "grades.score": {
//     "$gt": 90
//   }
// })

// db.restaurants.find({
//   "grades": {
//     "$elemMatch": { "score":
//       {"$gt": 1, "$lt": 4}
//     }
//   }
// })

// db.restaurants.find({
//   "address": {
//     "$elemMatch": { 
//       "coord": {"$lt": -95.754168}
//     }
//   }
// })

db.restaurants.find({
  "address.coord": {
    "$lt": -95
  }
  
})

db.restaturants.find({
  "$and": [
    {
      "cuisine": {
        "$ne": "American"
      }
    },
    {
      "grades.score": {
        "$gt": 70
      }
    }
  ]
})

//
db.employees.find(
  {
    hire_date: { 
      $gte: ISODate("2020-01-01T00:00:00Z"), 
      $lte: ISODate("2020-03-31T23:59:59Z") 
    },
    performance_rating: { $gt: 4.5 }
  },
  {
    _id: 0,
    name: 1,
    hire_date: 1,
    performance_rating: 1,
    "address.city": 1
  }
).sort({ performance_rating: -1 });

// Find all employees hired in 2021 with any certification containing "Service". RETURN: all fields EXCEPT performance_rating and store_id sorted by hire_date ascending
db.employees.find(
  {
    hire_date: {
      $gte: ISODate("2021-01-01T00:00:00Z"),
      $lte: ISODate("2021-12-31T23:59:59Z")
    },
    certifications: { $regex: "Service", $options: "i" }
  },
  {
    performance_rating: 0,
    store_id: 0
  }
).sort({ hire_date: 1 });

// Find employees whose performance rating is between 4.5 and 4.8 inclusive. RETURN: name, position, performance_rating fields only, sorted by performance_rating descending
db.employees.find(
  {
    performance_rating: { $gte: 4.5, $lte: 4.8 }
  },
  {
    _id: 0,
    name: 1,
    position: 1,
    performance_rating: 1
  }
).sort({ performance_rating: -1 });

// For stores with rating >= 4.6, add "premium_service" to the features array (if not exists) and add field "premium_since" with value "2024-01".
db.stores.updateMany(
  { rating: { $gte: 4.6 } },
  {
    $addToSet: { features: "premium_service" },
    $set: { premium_since: "2024-01" }
  }
);

// Remove "Barista Level 1" certification for Tomas Prochazka
db.employees.updateOne(
  { name: "Tomas Prochazka" },
  { $pull: { certifications: "Barista Level 1" } }
);

// Output the names of the stores that have at least one employee hired on or after January 1, 2023, sorted alphabetically by store name in ascending order.
db.stores.aggregate([
  {
    $lookup: {
      from: "employees",
      localField: "staff",
      foreignField: "_id",
      as: "employees"
    }
  },
  {
    $match: {
      "employees.hire_date": { $gte: ISODate("2023-01-01T00:00:00Z") }
    }
  },
  {
    $project: {
      _id: 0,
      name: 1
    }
  },
  {
    $sort: { name: 1 }
  }
]);

// Find stores open on weekends at or after 8:00 with wifi. RETURN: all fields EXCEPT staff and manager_id sorted by name ascending
db.stores.find(
  {
    "hours.saturday": { $gte: "08:00" },
    "hours.sunday": { $gte: "08:00" },
    features: "wifi"
  },
  {
    staff: 0,
    manager_id: 0
  }
).sort({ name: 1 });

// Calculate manager efficiency metrics comparing store performance and staff ratings. RETURN: manager_name, avg_store_rating (rounded to 2 decimals), avg_staff_performance (rounded to 2 decimals), manager_performance, store_count, sorted by avg_store_rating descending and manager_name ascending
// I have this incorrectly
db.stores.aggregate([
  {
    $lookup: {
      from: "employees",
      localField: "staff",
      foreignField: "_id",
      as: "staff_info"
    }
  },
  {
    $group: {
      _id: "$manager_id",
      manager_name: { $first: "$manager_id" },
      avg_store_rating: { $avg: "$rating" },
      avg_staff_performance: { $avg: { $avg: "$staff_info.performance_rating" } },
      store_count: { $sum: 1 }
    }
  },
  {
    $lookup: {
      from: "employees",
      localField: "_id",
      foreignField: "_id",
      as: "manager_info"
    }
  },
  {
    $addFields: {
      manager_name: { $arrayElemAt: ["$manager_info.name", 0] },
      manager_performance: { $avg: ["$avg_store_rating", "$avg_staff_performance"] }
    }
  },
  {
    $project: {
      _id: 0,
      manager_name: 1,
      avg_store_rating: { $round: ["$avg_store_rating", 2] },
      avg_staff_performance: { $round: ["$avg_staff_performance", 2] },
      manager_performance: { $round: ["$manager_performance", 2] },
      store_count: 1
    }
  },
  {
    $sort: { avg_store_rating: -1, manager_name: 1 }
  }
]);