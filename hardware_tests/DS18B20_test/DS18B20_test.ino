#include <OneWireNg_CurrentPlatform.h>

#define SENSOR_PIN 4

OneWireNg_CurrentPlatform ow(SENSOR_PIN, true);

bool readTemperature(float &temperatureC) {
  uint8_t data[9];w3xdi4yte6ea

  // Reset the 1-Wire bus
  ow.reset();

  // Skip ROM: we're talking to one sensor
  ow.writeByte(0xCC);

  // Start temperature conversion
  ow.writeByte(0x44);

  // Wait for conversion
  delay(800);

  // Reset the bus again
  ow.reset();

  // Skip ROM
  ow.writeByte(0xCC);

  // Read scratchpad
  ow.writeByte(0xBE);

  for (int i = 0; i < 9; i++) {
    data[i] = ow.readByte();
  }

  // Check CRC
  uint8_t crc = ow.crc8(data, 8);

  if (crc != data[8]) {
    return false;
  }

  // Combine the two temperature bytes
  int16_t rawTemperature = (data[1] << 8) | data[0];

  temperatureC = rawTemperature / 16.0;

  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println("ESP32-C5 DS18B20 TEST");
  Serial.println("================================");
}

void loop() {
  float temperatureC;

  if (readTemperature(temperatureC)) {
    Serial.print("Temperature: ");
    Serial.print(temperatureC, 2);
    Serial.println(" °C");
  } else {
    Serial.println("ERROR: Could not read DS18B20!");
  }

  delay(1000);
}