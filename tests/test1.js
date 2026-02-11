// Task: create the specified collection and insert documents into it that contain all the data listed in the examples. 
// Implement 5 queries. For each collection, insert one or more documents (in addition to the examples) so that each of the queries 2-5 returns a non-empty result.

// To test the queries, you can use the nosql.felk.cvut.cz server.

// Domain: A music agency that tracks concert events.
// Collection concerts:
// Each concert event stores information about the concert name and date/time, the main performer, the venue capacity, the ticket price, the projected audience size, and sponsor information (sponsor name and the amount invested). 
// The concert also includes details about the program, such as the number of supporting performances and optional special program items scheduled at a specific time (for example, a fireworks show). 
// Additional notes about VIP features may be stored.
// Example: 
//  concert “Indie Pulse” dated 05 May 2024, starring “Golden Strings”. 
//  Venue capacity 8,000, ticket price 45 USD, projected audience 6,200. Sponsor “LocalMusix” invests 20,000 USD. The event has 1 supporting performance and includes a meet-and-greet at 17:00.

// Queries:
// Query 1:
// The agency changed the program for “Indie Pulse”. 
// Decrease the ticket price by 5 USD, 
// increase the sponsor investment by 3,000 USD, 
// change the number of supporting performances from 1 to 2, 
// and remove the meet-and-greet at 17:00.

// Query 2:
// Find all concerts with projected audience at least 6,000. 
// Sort by projected audience from higher to lower. 
// Output the concert name, the concert date/time, and the projected audience. 
// Output at most 10 results starting from the first one.
// Query 3:
// Find all concerts that satisfy at least one of the following: the venue capacity is below 10,000, or there is a special program item scheduled at 17:00. 
// Sort by concert date/time from older to newer. 
// Output the concert name, the concert date/time, and only the matching special-program details. 
// Output at most 10 results starting from the first one.
// Query 4:
// For each concert, calculate an “attendance ratio” as projected audience / venue capacity. 
// Sort by attendance ratio from higher to lower. 
// Output the concert name, the sponsor name, and the attendance ratio. 
// Output at most 10 results.
// Query 5:
// For each sponsor, calculate the number of concerts sponsored and the average ticket price of those concerts. 
// Sort by the number of concerts from higher to lower. 
// Output the sponsor name, the number of concerts, and the average ticket price. 
// Output at most 10 results.

// All date/time fields must be stored as MongoDB BSON Date values (not strings). Input dates must be provided in ISO 8601 format. Store timestamps in UTC.
// For date-only fields, store them as BSON Date at 00:00:00Z. All filtering by month/year must be implemented via date ranges (start inclusive, end exclusive).

// event:
// - name “Indie Pulse”
// - date 05 May 2024
// - performer_main “Golden Strings”
// - capacity 8,000
// - ticket_price 45
// - the_projected_audience_size 6,200
// - sponsors (array of objects)
// --- sponsor name “LocalMusix”
// --- amount invested 20,000
// --- ...
// - program (object)
// --- number of supporting performances 1
// --- items (array of objects)
// ------ name
// ------ time
// - VIP
// 

// Inserting query
db.events.insertMany([
    {
        name: "Indie Pulse",
        date: ISODate("2024-05-05T00:00:00Z"),
        performer: "Goldern Strings",
        capacity: 8000,
        ticket_price: 45,
        projected_size: 6200,
        sponsor: {
                sponsor_name: "LocalMusix",
                invested: 20000
        },
        program: {
            num_supp_perf: 1,
            items: [
                { item_name: "meet-and-greet", item_time: "17:00" }
            ]
        }
    },
    {
        name: "Indie Pulse 2",
        date: ISODate("2024-06-05T00:00:00Z"),
        performer: "Goldern Strings 2",
        capacity: 8001,
        ticket_price: 55,
        projected_size: 6500,
        sponsor: {
                sponsor_name: "LocalMusix 2",
                invested: 2000
        },
        program: {
            num_supp_perf: 1,
            items: [
                { item_name: "meet-and-greet", item_time: "17:00" }
            ]
        }
    }
]);

// Query 1:
// The agency changed the program for “Indie Pulse”. 
// Decrease the ticket price by 5 USD, 
// increase the sponsor investment by 3,000 USD, 
// change the number of supporting performances from 1 to 2, 
// and remove the meet-and-greet at 17:00.
db.events.updateOne(
    {
        name: "Indie Pulse"
    },
    {
        $inc: {
            ticket_price: -5, "sponsors.invested": 3000 
        },
        $set: {
            "program.num_supp_perf": 2
        },
        $pull: {
            "program.items": {
                item_name: "meet-and-greet", item_time: "17:00"
            }
        }
    }
)

// Query 2:
// Find all concerts with projected audience at least 6,000. 
// Sort by projected audience from higher to lower. 
// Output the concert name, the concert date/time, and the projected audience. 
// Output at most 10 results starting from the first one.
db.events.find({
    projected_size: {
        $gte: 6000
    }
}, {
    "name": 1,
    "date": 1,
    "projected_size": 1,
    "_id": 0
}).sort({projected_size: -1}).limit(10);

// Query 3:
// Find all concerts that satisfy at least one of the following: the venue capacity is below 10,000, or there is a special program item scheduled at 17:00. 
// Sort by concert date/time from older to newer. 
// Output the concert name, the concert date/time, and only the matching special-program details. 
// Output at most 10 results starting from the first one.
db.events.find({
    $or: [
        { capacity: {$lt: 10000 } },
        { "program.items.item_time": "17:00" }
    ]
},
{
    "name": 1,
    "date": 1,
    "program.items.item_name": 1,
    "_id": 0
}).sort({date: 1}).limit(10);

// Query 4:
// For each concert, calculate an “attendance ratio” as projected audience / venue capacity. 
// Sort by attendance ratio from higher to lower. 
// Output the concert name, the sponsor name, and the attendance ratio. 
// Output at most 10 results.
db.events.aggregate([
    {
        $addFields: {
            ratio: {
                $divide: ["$projected_size", "$capacity"]
            }
        }
    },
    {
        $sort: { "ratio": -1 }
    },
    {
        $project: {
            "name": 1,
            "sponsor.sponsor_name": 1,
            "ratio": 1
        }
    },
    {
        $limit: 10
    }
]);

// Query 5:
// For each sponsor, calculate the number of concerts sponsored and the average ticket price of those concerts. 
// Sort by the number of concerts from higher to lower. 
// Output the sponsor name, the number of concerts, and the average ticket price. 
// Output at most 10 results.
db.events.aggregate([
    {
        $group: {
            "_id": "$sponsor.sponsor_name",
            "concert_count": { $sum: 1 },
            "avg_ticker_price": { $avg: "$ticket_price" }
        }
    },
    {
        $sort: { "concert_count": -1 }
    },
    {
        $project: {
            "_id": 1,
            "concert_count": 1,
            "avg_ticker_price": 1
        }
    },
    {
        $limit: 10
    }
]);