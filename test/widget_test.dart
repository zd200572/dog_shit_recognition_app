import 'package:dog_shit_detector/main.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('app widget can be constructed', () {
    expect(const DogShitDetectorApp(), isA<StatelessWidget>());
  });
}
