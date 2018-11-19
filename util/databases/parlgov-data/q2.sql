-- Winners

SET SEARCH_PATH TO parlgov;
drop table if exists q2 cascade;

-- You must not change this table definition.

create table q2(
countryName VARCHaR(100),
partyName VARCHaR(100),
partyFamily VARCHaR(100),
wonElections INT,
mostRecentlyWonElectionId INT,
mostRecentlyWonElectionYear INT
);


-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
-- Select all elections in the valid range
DROP VIEW IF EXISTS CrossedElections CASCADE;
DROP VIEW IF EXISTS Party2Win CASCADE;
DROP VIEW IF EXISTS QuantityParties CASCADE;
DROP VIEW IF EXISTS CountryMeanWin CASCADE;
DROP VIEW IF EXISTS Satisfy1 CASCADE;
DROP VIEW IF EXISTS Satisfy2 CASCADE;
DROP VIEW IF EXISTS Satisfy3 CASCADE;

-- Full joined table with election_id, year, country_id, party_id, and some attributes for the votes
CREATE OR REPLACE VIEW CrossedElections AS
SELECT ERange.*, party_id, votes
FROM (SELECT id, country_id,  (date_part('year', e_date)::INT) AS year
      FROM Election
     ) AS ERange JOIN Election_result AS ER ON Erange.id = ER.election_id
WHERE votes IS NOT NULL;

-- coutry_id, party_id, times this party has won elections
CREATE OR REPLACE VIEW Party2Win AS
SELECT country_id, party_id, COUNT(*) as timesWon
FROM CrossedElections as CEParent
WHERE votes = (SELECT MAX(votes)
               FROM CrossedElections as CE
               WHERE CEParent.id=CE.id)
GROUP BY country_id, party_id;

/* -- All parties which satisfy the constraint: Here the average is only from parties that have won by the winning parties
CREATE OR REPLACE VIEW Satisfy1 AS
SELECT *
FROM Party2Win as Parent
WHERE timesWon > (SELECT 3*AVG(timesWon) as meanWin
FROM Party2Win as Child
WHERE Child.country_id = Parent.country_id
GROUP BY Child.country_id)
ORDER BY party_id; */

-- Hold  country_id | party_id | timeswon | qtdparties
CREATE OR REPLACE VIEW QuantityParties AS
SELECT *
FROM Party2Win JOIN (SELECT Country.id as country_id, count(*) AS qtdParties
FROM Country JOIN Party ON Country.id = Party.country_id
GROUP BY Country.id) as CC USING(country_id);

CREATE OR REPLACE VIEW CountryMeanWin AS
SELECT country_id, 3*(SUM(timesWon)/AVG(qtdParties)) as meanWin
FROM QuantityParties
GROUP BY country_id;

CREATE OR REPLACE VIEW Satisfy1 AS
SELECT *
FROM Party2Win as Parent
WHERE timesWon > (SELECT meanWin
FROM CountryMeanWin as Child
WHERE Child.country_id = Parent.country_id)
ORDER BY party_id;


CREATE OR REPLACE VIEW Satisfy2 AS
SELECT party_id, Satisfy1.country_id, id, year, timesWon as wonElections
FROM (SELECT t1.id, country_id, party_id, year, topv
        FROM (SELECT Election.id,(date_part('year', e_date)::INT) AS year,  MAX(votes) as topv
              FROM Election JOIN Election_result as ER ON Election.id = ER.election_id
              GROUP BY Election.id) as t1
        JOIN
             (SELECT E.id, party_id, votes, E.country_id
              FROM Election_result as ER JOIN Election as E ON E.id = ER.election_id) as t2
        ON t1.id = t2.id
      WHERE t2.votes = t1.topv) as tempMax
    JOIN
    Satisfy1
    USING(party_id);

CREATE OR REPLACE VIEW Satisfy3 AS
SELECT country_id, party_id, wonElections, id as mostRecentlyWonElectionId, year as mostRecentlyWonElectionYear
FROM Satisfy2 as Parent
WHERE year = (SELECT MAX(year)
               FROM Satisfy2 as Child
               WHERE Parent.party_id = Child.party_id AND Parent.country_id = Child.country_id);

CREATE OR REPLACE VIEW Answer2 AS
SELECT Country.name AS countryName, Party.name AS partyName, Party_family.family as partyFamily, wonElections, mostRecentlyWonElectionId, mostRecentlyWonElectionYear
FROM Satisfy3 JOIN Country ON country_id = Country.id
              JOIN Party ON party_id = Party.id
              LEFT JOIN Party_family USING(party_id);

INSERT INTO q2
SELECT *
FROM Answer2;