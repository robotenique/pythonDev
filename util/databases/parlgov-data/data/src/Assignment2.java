import org.omg.CORBA.INTERNAL;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

// If you are looking for Java data structures, these are highly useful.
// Remember that an important part of your mark is for doing as much in SQL (not Java) as you can.
// Solutions that use only or mostly Java will not receive a high mark.
//import java.util.ArrayList;
//import java.util.Map;
//import java.util.HashMap;
//import java.util.Set;
//import java.util.HashSet;
public class Assignment2 extends JDBCSubmission {
    public Assignment2() throws ClassNotFoundException {

        Class.forName("org.postgresql.Driver");
    }

    @Override
    public boolean connectDB(String url, String username, String password) {
        try {
            super.connection = DriverManager.getConnection(url, username, password);
        }
        catch (SQLException e) {
            return false;
        }
        return true;
    }

    @Override
    public boolean disconnectDB() {
        if (super.connection == null)
            return false;
        try {
            super.connection.close();
        } catch (SQLException e) {
            return false;
        }
        return true;
    }

    @Override
    public ElectionCabinetResult electionSequence(String countryName) {
        PreparedStatement temp;
        ResultSet rsTemp;
        int country_id;
        try {
            temp = super.connection.prepareStatement("SELECT id " +
                                             "FROM Country " +
                                             "WHERE name = ?;");
             temp.setString(1, countryName);
             rsTemp = temp.executeQuery();

             if (rsTemp.next())
                 country_id = rsTemp.getInt("id");
             else // No country available, return empty;
                 return new ElectionCabinetResult(new ArrayList<>(), new ArrayList<>());

            temp = super.connection.prepareStatement("SELECT election_id, id as cabinet_id " +
                                            "FROM Cabinet " +
                                            "WHERE election_id IN (SELECT id as election_id " +
                                                                  "FROM Election " +
                                                                  "WHERE country_id = ?) " +
                                            "ORDER BY election_id; ");
            temp.setInt(1, country_id);
            rsTemp = temp.executeQuery();
            if (!rsTemp.isBeforeFirst())
               return null; // There are no available cabinets from this country
            List<Integer> eids = new ArrayList<>();
            List<Integer> cids = new ArrayList<>();
            while (rsTemp.next()) {
                eids.add(rsTemp.getInt("election_id"));
                cids.add(rsTemp.getInt("cabinet_id"));
            }
            return new ElectionCabinetResult(eids, cids);

        } catch (SQLException e) {
            e.printStackTrace();
        }
        return new ElectionCabinetResult(new ArrayList<>(), new ArrayList<>());
    }

    @Override
    public List<Integer> findSimilarPoliticians(Integer politicianName, Float threshold) {
        // Implement this method!
        PreparedStatement temp;
        ResultSet rsTemp;
        int politician_id = politicianName;
        String politician_fulltext;
        String desc;
        String comm;
        try {
            temp = super.connection.prepareStatement("SELECT id, description, comment " +
                                            "FROM politician_president " +
                                            "WHERE id = ?;");
            temp.setInt(1, politician_id);
            rsTemp = temp.executeQuery();
            if (rsTemp.next()) {
                desc = rsTemp.getString("description");
                comm = rsTemp.getString("comment");
                politician_fulltext = desc + " " + comm;
            }
            else // No politician available, return empty;
                return new ArrayList<>();
            temp = super.connection.prepareStatement("SELECT id,  description, comment " +
                                           "FROM politician_president " +
                                           "WHERE id <> ? ;");
            temp.setInt(1, politician_id);
            rsTemp = temp.executeQuery();
            List<Integer> final_pol = new ArrayList<>();
            while (rsTemp.next()) {
                desc = rsTemp.getString("description");
                comm = rsTemp.getString("comment");
                String curr_fulltext = desc + " " + comm;
                System.out.println(curr_fulltext);
                if (similarity(politician_fulltext, curr_fulltext) > threshold)
                    final_pol.add(rsTemp.getInt("id"));
            }
            return final_pol;
        } catch (SQLException e) {
            e.printStackTrace();
        }

        return new ArrayList<>();
    }


    public static void main(String[] args) {
        // You can put testing code in here. It will not affect our autotester.
        String url = "jdbc:postgresql://localhost:5432/parlgov";
        String username = "postgres";
        String password = "postgres";
        Assignment2 a2;
        try {
             a2 = new Assignment2();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            return;
        }
        System.out.println("Hello");
        a2.connectDB(url, username, password);
        System.out.println(a2.electionSequence("France"));
        System.out.println(a2.findSimilarPoliticians(148, 0.1F));
        a2.disconnectDB();
    }

}

