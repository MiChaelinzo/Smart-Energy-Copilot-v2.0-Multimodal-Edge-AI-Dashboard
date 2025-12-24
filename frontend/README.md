# Smart Energy Copilot v2.0 - Frontend

This is the React-based frontend for the Smart Energy Copilot v2.0 dashboard application.

## Features

- **Responsive Dashboard**: Auto-generated, responsive grid layout that adapts to different screen sizes
- **Interactive Visualizations**: Charts and graphs with zoom, pan, tooltip, and other interactive features
- **Real-time Updates**: WebSocket integration for live data updates
- **Energy Overview**: Current consumption, cost tracking, and trend analysis
- **AI Recommendations**: Display and management of AI-generated optimization recommendations
- **Device Monitoring**: Real-time status and consumption monitoring for IoT devices
- **Cost Analysis**: Breakdown of energy costs with trends and savings potential

## Technology Stack

- **React 18** with TypeScript
- **Material-UI (MUI)** for component library and theming
- **Recharts** for interactive data visualizations
- **React Grid Layout** for responsive, draggable dashboard widgets
- **Socket.IO Client** for real-time WebSocket communication
- **Axios** for HTTP API calls
- **Date-fns** for date formatting and manipulation

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend API server running on port 8000

### Installation

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

The application will open at `http://localhost:3000` and proxy API calls to `http://localhost:8000`.

### Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Architecture

### Component Structure

```
src/
├── components/
│   └── widgets/           # Dashboard widget components
├── contexts/              # React context providers
├── pages/                 # Page components
├── App.tsx               # Main application component
└── index.tsx             # Application entry point
```

### Key Components

- **Dashboard**: Main dashboard page with responsive grid layout
- **WebSocketContext**: Manages real-time WebSocket connections
- **DashboardContext**: Manages dashboard state and API calls
- **Widget Components**: Individual dashboard widgets (Energy Overview, Charts, etc.)

### State Management

The application uses React Context for state management:

- **WebSocketContext**: Handles real-time communication with the backend
- **DashboardContext**: Manages energy data, recommendations, and device information

### Responsive Design

The dashboard uses a responsive grid system that adapts to different screen sizes:

- **Desktop (1200px+)**: 12-column grid with full widget layout
- **Tablet (768-1199px)**: 12-column grid with adjusted widget sizes
- **Mobile (480-767px)**: 6-column grid with stacked widgets
- **Small Mobile (<480px)**: 4-column grid with minimal layout

## API Integration

The frontend communicates with the backend through:

- **REST API**: For data retrieval and updates (`/api/dashboard/*`)
- **WebSocket**: For real-time updates (`/ws`)

### API Endpoints Used

- `GET /api/dashboard/energy-data` - Retrieve energy consumption data
- `GET /api/dashboard/recommendations` - Get AI recommendations
- `PUT /api/dashboard/recommendations/{id}` - Update recommendation status
- `GET /api/dashboard/config` - Get dashboard configuration
- `PUT /api/dashboard/config` - Update dashboard configuration

## Features Implementation

### Interactive Visualizations

Charts support multiple interaction modes:
- **Zoom and Pan**: Mouse wheel and drag interactions
- **Tooltips**: Hover for detailed information
- **Chart Types**: Line, area, and bar charts
- **Time Ranges**: 24h, 7d, 30d views
- **Real-time Updates**: Automatic data refresh

### Real-time Updates

WebSocket integration provides:
- Live energy consumption updates
- Recommendation status changes
- Device status monitoring
- Connection status indicators

### Responsive Grid Layout

Dashboard widgets are:
- **Draggable**: Users can rearrange widgets
- **Resizable**: Widgets can be resized within constraints
- **Responsive**: Layout adapts to screen size
- **Persistent**: Layout preferences can be saved (future feature)

## Development

### Available Scripts

- `npm start` - Start development server
- `npm test` - Run test suite
- `npm run build` - Build for production
- `npm run eject` - Eject from Create React App (not recommended)

### Code Style

The project uses:
- TypeScript for type safety
- ESLint for code linting
- Prettier for code formatting (configured in package.json)

### Testing

Tests are written using:
- Jest for test runner
- React Testing Library for component testing
- Property-based tests for widget functionality

## Deployment

The frontend can be deployed as static files after building:

1. Build the application: `npm run build`
2. Serve the `build` folder using any static file server
3. Ensure the backend API is accessible at the configured proxy URL

For production deployment, consider:
- Setting up proper environment variables
- Configuring HTTPS
- Setting up CDN for static assets
- Implementing proper error boundaries and monitoring