-- VoteRange

SET SEARCH_PATH TO parlgov;
drop table if exists q1 cascade;

-- You must not change this table definition.

create table q1(
year INT,
countryName VARCHAR(50),
voteRange VARCHAR(20),
partyName VARCHAR(100)
);


-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
DROP VIEW IF EXISTS ElectionsInRange CASCADE;
DROP VIEW IF EXISTS CrossedElections CASCADE;
DROP VIEW IF EXISTS MultipleInstances CASCADE;
DROP VIEW IF EXISTS OnlyOneElec CASCADE;
DROP VIEW IF EXISTS CalculatedPairs;
DROP VIEW IF EXISTS Temp1;
DROP VIEW IF EXISTS Answer1;

-- Select all elections in the valid range
CREATE OR REPLACE VIEW ElectionsInRange AS
SELECT id, country_id,  (date_part('year', e_date)::INT) AS e_year, votes_valid
FROM Election
WHERE (date_part('year', e_date)::INT) >= 1996 AND
      (date_part('year', e_date)::INT) <= 2016 AND
      votes_valid IS NOT NULL;

-- Full joined table with election_id, year, country_id, party_id, and some attributes for the votes
CREATE OR REPLACE VIEW CrossedElections AS
SELECT ERange.*, party_id, votes, (votes/(cast(votes_valid as numeric)))*100 as party_range
FROM ElectionsInRange AS ERange JOIN Election_result AS ER ON Erange.id = ER.election_id
WHERE votes IS NOT NULL AND
      (votes/(cast(votes_valid as numeric)))*100 <> 0.0;

-- Get important info
CREATE OR REPLACE VIEW CalculatedPairs AS
SELECT country_id, e_year as year, party_id,  AVG(party_range) as party_range
FROM CrossedElections
GROUP BY country_id, e_year, party_id;

-- Build all the ranges
CREATE OR REPLACE VIEW Temp1 AS
(SELECT country_id, year, party_id, party_range, '(0-5]' as voteRange
FROM CalculatedPairs
WHERE party_range > 0.0 AND party_range <= 5.0)
UNION
(SELECT country_id, year, party_id, party_range, '(5-10]' as voteRange
FROM CalculatedPairs
WHERE party_range > 5.0 AND party_range <= 10.0)
UNION
(SELECT country_id, year, party_id, party_range, '(10-20]' as voteRange
FROM CalculatedPairs
WHERE party_range > 10.0 AND party_range <= 20.0)
UNION
(SELECT country_id, year, party_id, party_range, '(20-30]' as voteRange
FROM CalculatedPairs
WHERE party_range > 20.0 AND party_range <= 30.0)
UNION
(SELECT country_id, year, party_id, party_range, '(30-40]' as voteRange
FROM CalculatedPairs
WHERE party_range > 30.0 AND party_range <= 40.0)
UNION
(SELECT country_id, year, party_id, party_range, '(40-100]' as voteRange
FROM CalculatedPairs
WHERE party_range > 40.0);


CREATE OR REPLACE VIEW Answer1 AS
SELECT year, Country.name AS countryName, Party.name_short AS partyName, voteRange
FROM Temp1 JOIN Country ON Temp1.country_id = Country.id
           JOIN Party   ON Temp1.party_id = Party.id;

INSERT INTO q1
SELECT *
FROM Answer1;