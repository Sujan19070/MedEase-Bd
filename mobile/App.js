import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import ChatScreen   from './screens/ChatScreen';
import SearchScreen from './screens/SearchScreen';
import AboutScreen  from './screens/AboutScreen';

const Tab = createBottomTabNavigator();

export const COLORS = {
  bg:      '#0A1628',
  surface: '#111E33',
  card:    '#162440',
  border:  '#1E3A5F',
  accent:  '#00D4FF',
  accent2: '#0EA5E9',
  green:   '#10B981',
  amber:   '#F59E0B',
  red:     '#F43F5E',
  white:   '#FFFFFF',
  light:   '#E2E8F0',
  muted:   '#64748B',
};

// ── Change this to your Desktop IP when on same WiFi ──────────
// For USB / localhost use: http://127.0.0.1:5000
export const API_BASE = 'http://192.168.0.148/24';  // ← change to your desktop IP

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="light" backgroundColor={COLORS.bg} />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerShown: false,
          tabBarStyle: {
            backgroundColor: COLORS.surface,
            borderTopColor:  COLORS.border,
            borderTopWidth:  1,
            height:          62,
            paddingBottom:   8,
            paddingTop:      6,
          },
          tabBarActiveTintColor:   COLORS.accent,
          tabBarInactiveTintColor: COLORS.muted,
          tabBarLabelStyle: { fontSize: 11, fontWeight: '600' },
          tabBarIcon: ({ focused, color, size }) => {
            const icons = {
              Chat:   focused ? 'chatbubbles'        : 'chatbubbles-outline',
              Search: focused ? 'search'             : 'search-outline',
              About:  focused ? 'information-circle' : 'information-circle-outline',
            };
            return <Ionicons name={icons[route.name]} size={22} color={color} />;
          },
        })}
      >
        <Tab.Screen name="Chat"   component={ChatScreen}   />
        <Tab.Screen name="Search" component={SearchScreen} />
        <Tab.Screen name="About"  component={AboutScreen}  />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
