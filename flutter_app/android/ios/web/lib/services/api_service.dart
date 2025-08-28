import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl = "http://10.0.2.2:5000"; // Android emulator → localhost

  Future<List<dynamic>> fetchLogs() async {
    final response = await http.get(Uri.parse('$baseUrl/logs/'));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Failed to load logs");
    }
  }

  Future<void> addLog(String emotion, String timestamp) async {
    final response = await http.post(
      Uri.parse('$baseUrl/logs/'),
      headers: {"Content-Type": "application/json"},
      body: json.encode({"emotion": emotion, "timestamp": timestamp}),
    );
    if (response.statusCode != 201) {
      throw Exception("Failed to save log");
    }
  }
}
