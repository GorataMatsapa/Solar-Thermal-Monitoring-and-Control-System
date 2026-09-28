// =====================================================
// MACROS & CONFIGURATION
// =====================================================
#define ENABLE_USER_AUTH
#define ENABLE_DATABASE

// =====================================================
// INCLUDES
// =====================================================
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <FirebaseClient.h>
#include <OneWireNg_CurrentPlatform.h>
#include <secrets.h>
// =====================================================
// Wi-Fi Configuration
// =====================================================
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// =====================================================
// Firebase Configuration
// =====================================================
#define API_KEY API_KEY
#define USER_EMAIL USER_EMAIL
#define USER_PASSWORD USER_PASSWORD
#define DATABASE_URL DATABASE_URL

// =====================================================
// Hardware Pin Definitions
// =====================================================
#define SENSOR_PIN 4
#define LED_PIN 2

OneWireNg_CurrentPlatform ow(SENSOR_PIN, true);

// =====================================================
// Firebase Objects
// =====================================================
WiFiClientSecure ssl_client_write;
WiFiClientSecure ssl_client_stream;

using AsyncClient = AsyncClientClass;

AsyncClient writeClient(ssl_client_write);
AsyncClient streamClient(ssl_client_stream);

UserAuth user_auth(API_KEY, USER_EMAIL, USER_PASSWORD);
FirebaseApp app;
RealtimeDatabase Database;

AsyncResult writeResult;
AsyncResult streamResult;

bool streamSubscribed = false;
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 10000; // 10 seconds

// =====================================================
// DS18B20 Reader
// =====================================================
bool readTemperature(float &temperatureC) {
  uint8_t data[9];
  ow.reset();
  ow.writeByte(0xCC);
  ow.writeByte(0x44);
  delay(800);

  ow.reset();
  ow.writeByte(0xCC);
  ow.writeByte(0xBE);

  for (int i = 0; i < 9; i++) {
    data[i] = ow.readByte();
  }

  if (ow.crc8(data, 8) != data[8]) return false;

  int16_t rawTemperature = (data[1] << 8) | data[0];
  temperatureC = rawTemperature / 16.0;
  return true;
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); // Default OFF

  delay(1000);
  Serial.println("\n======================================");
  Serial.println("STMaCS ACTIVE STREAM LISTENER TEST");
  Serial.println("======================================");

  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connected! IP: " + WiFi.localIP().toString());

  ssl_client_write.setInsecure();
  ssl_client_stream.setInsecure();

  initializeApp(writeClient, app, getAuth(user_auth), writeResult);
  app.getApp<RealtimeDatabase>(Database);
  Database.url(DATABASE_URL);
}

// =====================================================
// MAIN LOOP
// =====================================================
void loop() {
  // Keep internal operations active
  app.loop();

  if (!app.ready()) return;

  // ---------------------------------------------------
  // 1. Subscribe to Real-Time Stream (Once Auth is Ready)
  // ---------------------------------------------------
  if (!streamSubscribed) {
    Serial.println("Starting Realtime Stream on /stmacs/led_state...");
    Database.get(streamClient, "/stmacs/led_state", streamResult, true /* SSE */);
    streamSubscribed = true;
  }

  // ---------------------------------------------------
  // 2. Active Stream Event Handler
  // ---------------------------------------------------
  if (streamResult.isResult()) {
    if (streamResult.available()) {
      RealtimeDatabaseResult data = streamResult.to<RealtimeDatabaseResult>();
      
      String payload = data.to<String>();
      payload.toLowerCase();
      payload.replace("\"", ""); // Clean up escaped quotes
      payload.trim();

      Serial.print("Stream event received -> Raw value: ");
      Serial.println(payload);

      if (payload == "true" || payload == "1" || payload == "on") {
        digitalWrite(LED_PIN, HIGH);
        Serial.println(">>> LED STATUS: TURNED ON");
      } else if (payload == "false" || payload == "0" || payload == "off") {
        digitalWrite(LED_PIN, LOW);
        Serial.println(">>> LED STATUS: TURNED OFF");
      }
    }
  }

  // ---------------------------------------------------
  // 3. Outbound Telemetry (Non-Blocking)
  // ---------------------------------------------------
  if (millis() - lastSendTime >= sendInterval) {
    lastSendTime = millis();

    float tempC;
    if (readTemperature(tempC)) {
      Serial.print("Sending Temp: ");
      Serial.print(tempC);
      Serial.println(" °C");

      Database.set<float>(writeClient, "/stmacs/temperature", tempC, writeResult);
      Database.set<String>(writeClient, "/stmacs/status", "online", writeResult);
    }
  }
}