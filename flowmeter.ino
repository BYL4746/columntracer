#include <Wire.h>
const int ADDRESS = 0x40;
const float SCALE_FACTOR_FLOW = 256.0;
const float OFFSET = 0.0;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Wire.begin();
  delay(100);
  
  
  
}

void loop() {
  // put your main code here, to run repeatedly:
  uint16_t flow_value;
  int16_t signed_flow_value;
  float scaled_flow_value;
  Wire.beginTransmission(ADDRESS);
  Wire.write(0x10);
  Wire.write(0x00);
  //Wire.write(FlowMeasure);
  //Wire.write(TempMeasure)

  Wire.endTransmission();
  delay(100);
  Wire.requestFrom(ADDRESS,2);
  flow_value = Wire.read() << 8; //MSB
  flow_value |= Wire.read(); //LSB
  
  Serial.print("Flow= ");
  Serial.println(flow_value);
  
  signed_flow_value = (int16_t) flow_value;
  Serial.print("Signed Flow= ");
  Serial.print(signed_flow_value);
  
  scaled_flow_value = (((float) signed_flow_value) - OFFSET) / SCALE_FACTOR_FLOW;
  Serial.print(", scaled value: ");
  Serial.print(scaled_flow_value);
  Serial.println("");
  delay(1000);
}
