import java.sql.*;
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
    private Connection conn;
    public Assignment2() throws ClassNotFoundException {

        Class.forName("org.postgresql.Driver");
    }

    @Override
    public boolean connectDB(String url, String username, String password) {
        try {
            conn = DriverManager.getConnection(url, username, password);
        }
        catch (SQLException e) {
            return false;
        }
        return true;
    }

    @Override
    public boolean disconnectDB() {
        if (conn == null)
            return false;
        try {
            conn.close();
        } catch (SQLException e) {
            return false;
        }
        return true;
    }

    @Override
    public ElectionCabinetResult electionSequence(String countryName) {
        PreparedStatement temp;
        ResultSet rsTemp;
        "previous_parliament_election_id"
                "previous_ep_election_id"
        int country_id;
        try {
            PreparedStatement ps = conn.prepareStatement("SET SEARCH_PATH TO parlgov;");
            ps.execute();
             temp = conn.prepareStatement("SELECT id\n" +
                                 "           FROM Country\n" +
                                "            WHERE name = ?");
             temp.setString(1, countryName);
             rsTemp = temp.executeQuery();

             if (rsTemp.next())
                 country_id = rsTemp.getInt("id");
             else // No country available!
                 return null;

            /*

            SELECT *
            FROM Election
            WHERE country_id =
             */
            PreparedStatement stmt = conn.prepareStatement("SELECT * FROM Election where id=9");
            ResultSet rs = stmt.executeQuery();

            while (rs.next()) {
                int id = rs.getInt("id");
                int country_id_2 = rs.getInt("country_id");
                int votes_valid = rs.getInt("votes_valid");
                String e_type = rs.getString("e_type");
                System.out.println("id: "+id+" , "+country_id_2+", "+votes_valid+", "+e_type);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public List<Integer> findSimilarPoliticians(Integer politicianName, Float threshold) {
        // Implement this method!
        return null;
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
        a2.electionSequence("France");


        a2.disconnectDB();
    }

}

