// =====================================================
// MACROS & CONFIGURATION
// =====================================================
#define ENABLE_USER_AUTH
#define ENABLE_DATABASE
#define ENABLE_GSM_NETWORK
#define ENABLE_ESP_SSLCLIENT

// Map the SIM7670G architecture to its matching structural driver code
#define TINY_GSM_MODEM_SIM7600
#define TINY_GSM_RX_BUFFER 1024 

// =====================================================
// INCLUDES
// =====================================================
#include <Arduino.h>
#include <TinyGsmClient.h>       
#include <FirebaseClient.h>

#include <OneWireNg_CurrentPlatform.h>
#include <secrets.h>

// =====================================================
// Onboard Hardware Pins (Waveshare S3 SIM7670G Specific)
// =====================================================
#define PIN_TX      18
#define PIN_RX      17
#define PIN_PWRKEY  14

#define SENSOR_PIN  4
#define LED_PIN     2

// =====================================================
// Cellular Carrier Settings (FIXED: Added missing array brackets)
// =====================================================
const char apn[] = "internet.be";
const char gprsUser[] = "beinternet";
const char gprsPass[] = "1234";

// =====================================================
// Firebase Configuration
// =====================================================
#define API_KEY API_KEY
#define USER_EMAIL USER_EMAIL
#define USER_PASSWORD USER_PASSWORD
#define DATABASE_URL DATABASE_URL

// =====================================================
// =====================================================
// Hardware Instances
// =====================================================
HardwareSerial SerialAT(2); 
TinyGsm modem(SerialAT);

// Standard TinyGsmClient objects
TinyGsmClient cellular_client_write(modem, 0);
TinyGsmClient cellular_client_stream(modem, 1);

// --- ADD THIS LINE HERE: Ties Firebase to the GSM Modem Network Layer ---
GSMNetwork gsm_network(&modem, "", apn, gprsUser, gprsPass); 

OneWireNg_CurrentPlatform ow(SENSOR_PIN, true);

// =====================================================
// Firebase Objects
// =====================================================
ESP_SSLClient ssl_client;
ESP_SSLClient stream_ssl_client;

using AsyncClient = AsyncClientClass;


// CHANGED: Explicitly fetch the internal network configuration block
AsyncClient writeClient(ssl_client, network.getNetwork());
AsyncClient streamClient(stream_ssl_client, network.getNetwork());

// The rest of your definitions continue normally below...
UserAuth user_auth(API_KEY, USER_EMAIL, USER_PASSWORD, 3000);


FirebaseApp app;
RealtimeDatabase Database;

AsyncResult writeResult;
AsyncResult streamResult;

bool streamSubscribed = false;
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 10000; 

// =====================================================
// DS18B20 Reader (FIXED: Restored data[9] array bounds matching original)
// =====================================================
bool readTemperature(float &temperatureC) {
  uint8_t data[9];

  // Start temperature conversion
  ow.reset();
  ow.writeByte(0xCC);   // Skip ROM
  ow.writeByte(0x44);   // Convert T

  delay(800);

  // Read temperature scratchpad
  ow.reset();
  ow.writeByte(0xCC);   // Skip ROM
  ow.writeByte(0xBE);   // Read Scratchpad

  for (int i = 0; i < 9; i++) {
    data[i] = ow.readByte();
  }

  // Check CRC
  if (ow.crc8(data, 8) != data[8]) {
    Serial.println("Temperature sensor CRC FAILED");
    return false;
  }

  int16_t rawTemperature = (data[1] << 8) | data[0];
  temperatureC = rawTemperature / 16.0;

  return true;
}

// =====================================================
// CELLULAR INITIALIZATION ROUTINE
// =====================================================
void initializeCellular() {
  pinMode(PIN_PWRKEY, OUTPUT);
  digitalWrite(PIN_PWRKEY, HIGH);
  delay(300); 
  digitalWrite(PIN_PWRKEY, LOW);
  
  Serial.print("Waiting for modem to boot up...");
  delay(5000); 
  Serial.println(" Done.");

  SerialAT.begin(115200, SERIAL_8N1, PIN_RX, PIN_TX);

  Serial.println("Initializing modem...");
  if (!modem.restart()) {
    Serial.println("Failed to restart modem, trying fallback initialization...");
    if (!modem.init()) {
      Serial.println("Critical Error: Modem communication failed.");
      while (true); 
    }
  }

  Serial.print("Waiting for cellular network registration...");
  if (!modem.waitForNetwork(60000L)) {
    Serial.println(" Fail. Check antenna alignment.");
    while (true);
  }
  Serial.println(" Registered successfully!");

  Serial.print("Connecting to APN: ");
  Serial.println(apn);
  if (!modem.gprsConnect(apn, gprsUser, gprsPass)) {
    Serial.println(" GPRS Connection Failed.");
    while (true);
  }
  Serial.println("Cellular Network Online!");
}
// =====================================================
// FIREBASE AUTHENTICATION DEBUG
// =====================================================
void authDebugPrint(AsyncResult &aResult) {

  if (aResult.isEvent()) {
    Serial.print("Firebase Event: ");
    Serial.println(aResult.eventLog().message().c_str());
  }

  if (aResult.isDebug()) {
    Serial.print("Firebase Debug: ");
    Serial.println(aResult.debug());
  }

  if (aResult.isError()) {
    Serial.print("Firebase Error: ");
    Serial.println(aResult.error().message().c_str());
  }
}
// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 

  delay(1000);
  Serial.println("\n======================================");
  Serial.println("STMaCS ACTIVE CELLULAR STREAM TEST");
  Serial.println("======================================");

  // Bring the cellular internet stack online first
   initializeCellular();

    Serial.println("Configuring SSL transport...");

  ssl_client.setInsecure();
  ssl_client.setDebugLevel(1);
  ssl_client.setTimeout(15000); // <-- ADD THIS: Give it 15 seconds for the SSL Handshake
  ssl_client.setBufferSizes(2048, 1024);
  ssl_client.setClient(&cellular_client_write);

  stream_ssl_client.setInsecure();
  stream_ssl_client.setDebugLevel(1);
  stream_ssl_client.setTimeout(15000); // <-- ADD THIS: Same for the stream client
  stream_ssl_client.setBufferSizes(2048, 1024);
  stream_ssl_client.setClient(&cellular_client_stream);


  Serial.println("SSL transport ready.");

  Serial.println("Initializing Firebase app...");

initializeApp(
  writeClient,
  app,
  getAuth(user_auth),
  authDebugPrint,
  "authTask"
);

 app.getApp<RealtimeDatabase>(Database);
 Database.url(DATABASE_URL);

 Serial.println("Firebase initialization requested.");
  
}

// =====================================================
// MAIN LOOP
// =====================================================
void loop() {
  app.loop();

  // Keep network connection secure and active
  if (!modem.isGprsConnected()) {
    Serial.println("Cellular connection dropped! Reconnecting...");
    modem.gprsConnect(apn, gprsUser, gprsPass);
    return;
  }

if (!app.ready()) {
  Serial.println("Firebase app not ready yet...");
  delay(1000);
  return;
}

  // 1. Subscribe to Real-Time Stream (Once Auth is Ready)
  if (!streamSubscribed) {
    Serial.println("Starting Realtime Stream on /stmacs/led_state...");
    Database.get(streamClient, "/stmacs/led_state", streamResult, true /* SSE */);
    streamSubscribed = true;
  }

  // 2. Active Stream Event Handler
  if (streamResult.isResult()) {
    if (streamResult.available()) {
      RealtimeDatabaseResult data = streamResult.to<RealtimeDatabaseResult>();
      
      String payload = data.to<String>();
      payload.toLowerCase();
      payload.replace("\"", ""); 
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

  // 3. Outbound Telemetry (Non-Blocking)
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
