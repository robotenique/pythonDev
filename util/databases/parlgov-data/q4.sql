-- Left-right

SET SEARCH_PATH TO parlgov;
drop table if exists q4 cascade;

-- You must not change this table definition.


CREATE TABLE q4(
        countryName VARCHAR(50),
        r0_2 INT,
        r2_4 INT,
        r4_6 INT,
        r6_8 INT,
        r8_10 INT
);

-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
-- Countries with NO parties at all should be included!
DROP VIEW IF EXISTS CountryParties CASCADE;
DROP VIEW IF EXISTS CPartyPos CASCADE;
DROP VIEW IF EXISTS AlmostFinal CASCADE;

CREATE OR REPLACE VIEW CountryParties AS
SELECT Country.id as country_id, Party.id as party_id
FROM Country LEFT JOIN Party on Country.id = Party.country_id;



-- Join with all the party positions. Even if a party has no position
CREATE OR REPLACE VIEW CPartyPos AS
SELECT CP.*, Party_position.left_right as partypos
FROM CountryParties as CP LEFT JOIN Party_position USING(party_id);

CREATE OR REPLACE VIEW AlmostFinal AS
SELECT country_id, coalesce(r0_2, 0) AS r0_2, coalesce(r2_4, 0) AS r2_4, coalesce(r4_6, 0) AS r4_6, coalesce(r6_8, 0) AS r6_8, coalesce(r8_10, 0) AS r8_10
FROM (SELECT country_id, count(*) as r0_2
      FROM CPartyPos
      WHERE partypos >= 0.0 AND partypos < 2.0
      GROUP BY country_id) as t1
         LEFT JOIN
            (SELECT country_id, count(*) as r2_4
            FROM CPartyPos
            WHERE partypos >= 2.0 AND partypos < 4.0
            GROUP BY country_id) as t2 USING(country_id)
         LEFT JOIN
            (SELECT country_id, count(*) as r4_6
            FROM CPartyPos
            WHERE partypos >= 4.0 AND partypos < 6.0
            GROUP BY country_id) as t4 USING(country_id)
         LEFT JOIN
            (SELECT country_id, count(*) as r6_8
            FROM CPartyPos
            WHERE partypos >= 6.0 AND partypos < 8.0
            GROUP BY country_id) as t5 USING(country_id)
         LEFT JOIN
            (SELECT country_id, count(*) as r8_10
            FROM CPartyPos
            WHERE partypos >= 8.0 AND partypos < 10.0
            GROUP BY country_id) as t6 USING(country_id);

CREATE OR REPLACE VIEW Answer4 AS
SELECT Country.name AS countryName, r0_2, r2_4, r4_6, r6_8, r8_10
FROM AlmostFinal JOIN Country ON country_id = Country.id;

INSERT INTO q4
SELECT *
FROM Answer4;