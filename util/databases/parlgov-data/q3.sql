-- Participate

SET SEARCH_PATH TO parlgov;
drop table if exists q3 cascade;

-- You must not change this table definition.

create table q3(
        countryName varchar(50),
        year int,
        participationRatio real
);

-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
DROP VIEW IF EXISTS CrossedElections CASCADE;
DROP VIEW IF EXISTS Valid1 CASCADE;
DROP VIEW IF EXISTS Valid2 CASCADE;
DROP VIEW IF EXISTS Valid3 CASCADE;
DROP VIEW IF EXISTS MoreThanOneElection CASCADE;
DROP VIEW IF EXISTS distCountries CASCADE;
DROP VIEW IF EXISTS ProblematicCountries CASCADE;
DROP VIEW IF EXISTS FullFiltered CASCADE;
DROP VIEW IF EXISTS FinalBaseTable CASCADE;
DROP VIEW IF EXISTS ValidCountriesAtFinalBaseTable CASCADE;
DROP VIEW IF EXISTS Answer3 CASCADE;


-- Table with id, country_id, year and participationRation
CREATE OR REPLACE VIEW CrossedElections AS
SELECT id, country_id,  (date_part('year', e_date)::INT) AS year, (COALESCE (votes_cast,0)/(cast(electorate as numeric))) AS participationRatio
FROM Election
WHERE  (date_part('year', e_date)::INT) >= 2001 AND
       (date_part('year', e_date)::INT) <= 2016;

-- Have one election in all of the years between 2001 and 2016, so it's valid
CREATE OR REPLACE VIEW Valid1 AS
SELECT country_id, year, participationRatio
FROM CrossedElections
WHERE country_id IN (SELECT country_id
                     FROM CrossedElections
                     GROUP BY country_id
                     HAVING count(*) = 1);

CREATE OR REPLACE VIEW MoreThanOneElection AS
(SELECT country_id, year, participationRatio FROM CrossedElections)
EXCEPT
(SELECT * FROM Valid1);

-- Pair country_id, year which have more than 1 election in that year (problematic!)
CREATE OR REPLACE VIEW ProblematicCountries AS
SELECT country_id, year
      FROM MoreThanOneElection
      GROUP BY country_id, year
      HAVING count(*) > 1;


-- Create a version of CrossedElections but with some corrections to the average of ProblematicCountries
CREATE OR REPLACE VIEW FullFiltered AS
(SELECT country_id, year, participationRatio
FROM MoreThanOneElection
WHERE (country_id, year) NOT IN (SELECT * FROM ProblematicCountries))
UNION
(SELECT country_id, year, AVG(participationRatio) as participationRatio
FROM ProblematicCountries JOIN MoreThanOneElection USING(country_id, year)
GROUP BY country_id, year);

-- After our filtering, maybe some countries are now with just one row!
CREATE OR REPLACE VIEW Valid2 AS
SELECT *
FROM FullFiltered
WHERE country_id IN (SELECT country_id
                     FROM FullFiltered
                     GROUP BY country_id
                     HAVING count(*) = 1);

CREATE OR REPLACE VIEW FinalBaseTable AS
(SELECT * FROM FullFiltered)
EXCEPT
(SELECT * FROM Valid2);

CREATE OR REPLACE VIEW ValidCountriesAtFinalBaseTable AS
(SELECT DISTINCT country_id
FROM FinalBaseTable)
EXCEPT
(SELECT DISTINCT t1.country_id as country_id
 FROM FinalBaseTable as t1 JOIN FinalBaseTable as t2
 ON t1.country_id = t2.country_id AND t1.year < t2.year AND t1.participationRatio > t2.participationRatio);


CREATE OR REPLACE VIEW Valid3 AS
SELECT country_id, year, participationRatio
FROM FinalBaseTable
WHERE country_id IN (SELECT * FROM ValidCountriesAtFinalBaseTable);

CREATE OR REPLACE VIEW Answer3 AS
(SELECT Country.name as countryName, year, participationRatio
 FROM Valid1 JOIN Country ON country_id = Country.id)
 UNION
(SELECT Country.name as countryName, year, participationRatio
 FROM Valid2 JOIN Country ON country_id = Country.id)
 UNION
(SELECT Country.name as countryName, year, participationRatio
 FROM Valid3 JOIN Country ON country_id = Country.id);

INSERT INTO q3
SELECT *
FROM Answer3;