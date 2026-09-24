import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://ozopcneaghprtmnfrokh.supabase.co'
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96b3BjbmVhZ2hwcnRtbmZyb2toIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM2NTE5NDMsImV4cCI6MjA4OTIyNzk0M30.K-Sh_vTb0h_7dIGE9pHoRXc4IbNTWaU5yZzjxYwdca8'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
